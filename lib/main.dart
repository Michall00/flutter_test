import 'dart:io';
import 'dart:typed_data';
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_crashlytics/firebase_crashlytics.dart';
import 'firebase_options.dart';
import 'utils/tensor_utils.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );
  await FirebaseCrashlytics.instance.setCrashlyticsCollectionEnabled(true);
  FlutterError.onError = FirebaseCrashlytics.instance.recordFlutterFatalError;
  FirebaseCrashlytics.instance.log("App started");
  FirebaseCrashlytics.instance
      .setCustomKey("app_start_time", DateTime.now().toIso8601String());

  runZonedGuarded(
    () => runApp(const MyApp()),
    (error, stackTrace) =>
        FirebaseCrashlytics.instance.recordError(error, stackTrace),
  );
}

final GlobalKey imageKey = GlobalKey();

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'MI-GAN Inpainting',
      theme: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: Colors.blue,
      ),
      home: ImagePickerPage(),
    );
  }
}

class ImagePickerPage extends StatefulWidget {
  @override
  State<ImagePickerPage> createState() => _ImagePickerPageState();
}

class _ImagePickerPageState extends State<ImagePickerPage> {
  File? _imageFile;
  img.Image? _maskImage;
  int? _imageWidth;
  int? _imageHeight;
  Uint8List? _segmentationMask;

  void _startInpaintingWithSnackBar() async {
    final messenger = ScaffoldMessenger.of(context);

    try {
      await _runInpainting();

      messenger.showSnackBar(
        const SnackBar(
          content: Text('✅ Inpainting zakończony!'),
          duration: Duration(seconds: 1),
        ),
      );
    } catch (e, stacktrace) {
      debugPrint("Błąd inpaintingu: $e");
      debugPrint("Stacktrace:\n$stacktrace");

      messenger.showSnackBar(
        SnackBar(
          content: Text('❌ Błąd: $e'),
          duration: const Duration(seconds: 3),
        ),
      );
    }
  }

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(source: ImageSource.gallery);
    if (picked != null) {
      final file = File(picked.path);
      final bytes = await file.readAsBytes();
      final decoded = img.decodeImage(bytes);

      if (decoded == null) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('❌ Nie udało się odczytać obrazu')),
        );
        return;
      }
      const targetSize = 1024;

      final resized = img.copyResize(
        decoded,
        width: targetSize,
        height: targetSize,
        interpolation: img.Interpolation.linear,
      );

      final resultBytes = Uint8List.fromList(img.encodePng(resized));
      final tempFile = await File(
              '${(await Directory.systemTemp.createTemp()).path}/input.png')
          .writeAsBytes(resultBytes);

      FirebaseCrashlytics.instance.log("Obraz wybrany przez użytkownika");
      FirebaseCrashlytics.instance.setCustomKey(
          "image_resolution", "${resized.width}x${resized.height}");

      setState(() {
        _imageFile = tempFile;
        _imageWidth = resized.width;
        _imageHeight = resized.height;
        _segmentationMask = null;
        _maskImage = img.Image(
            width: resized.width, height: resized.height, numChannels: 1)
          ..getBytes().fillRange(0, resized.width * resized.height, 255);
      });
    }
  }

  Future<void> _runInpainting() async {
    FirebaseCrashlytics.instance.log("🎨 Rozpoczęcie inpaintingu");
    final messenger = ScaffoldMessenger.of(context);
    if (_imageFile == null) {
      messenger.showSnackBar(
        const SnackBar(
          content: Text('Brak pliku obrazu!'),
          duration: Duration(seconds: 1),
        ),
      );
      return;
    }

    final bytes = await _imageFile!.readAsBytes();
    final originalImage = img.decodeImage(bytes)!;

    if (_segmentationMask != null) {
      _maskImage = img.decodeImage(_segmentationMask!)!;
    }

    final imageTensor = convertImageToUint8NCHW(originalImage);
    final maskTensor = convertMaskToUint8NCHW(_maskImage!);

    OrtEnv.instance.init();
    final modelData = await rootBundle.load('assets/migan.onnx');
    final session = OrtSession.fromBuffer(
      modelData.buffer.asUint8List(),
      OrtSessionOptions(),
    );

    final start = DateTime.now();
    final result = session.run(
      OrtRunOptions(),
      {'image': imageTensor, 'mask': maskTensor},
      ['result'],
    );
    final duration = DateTime.now().difference(start).inMilliseconds;
    imageTensor.release();
    maskTensor.release();
    session.release();
    FirebaseCrashlytics.instance.setCustomKey("inpaint_duration_ms", duration);

    final output = result[0]!.value as List;
    final imgOut = convertNCHWtoImage(output);
    FirebaseCrashlytics.instance.log("✅ MI-GAN inference zakończony sukcesem");
    FirebaseCrashlytics.instance
        .setCustomKey("result_size", "${output.length} px");

    final upscaledResult = img.copyResize(
      imgOut,
      width: _imageWidth!,
      height: _imageHeight!,
      interpolation: img.Interpolation.linear,
    );

    final resultBytes = Uint8List.fromList(img.encodeJpg(upscaledResult));

    final tempDir = await Directory.systemTemp.createTemp();
    final filePath = '${tempDir.path}/output.jpg';
    final resultFile = await File(filePath).writeAsBytes(resultBytes);

    setState(() {
      _imageFile = resultFile;
    });
  }

  Future<void> _runSegmentationFromClick(Offset point) async {
    FirebaseCrashlytics.instance.log(
        "🎯 Start segmentacji na pozycji: \${point.dx.toStringAsFixed(1)}, \${point.dy.toStringAsFixed(1)}");
    final bytes = await _imageFile!.readAsBytes();
    final image = img.decodeImage(bytes)!;
    final pixels = image.getBytes(order: img.ChannelOrder.rgb);
    final imgFloat = Float32List(pixels.length);
    for (int i = 0; i < pixels.length; i++) {
      imgFloat[i] = pixels[i].toDouble();
    }

    final encoderInput = OrtValueTensor.createTensorWithDataList(
        imgFloat, [image.height, image.width, 3]);
    final encoderData = await rootBundle.load('assets/encoder.onnx');
    final encoderSession = OrtSession.fromBuffer(
        encoderData.buffer.asUint8List(), OrtSessionOptions());
    final embeddings = encoderSession.run(
      OrtRunOptions(),
      {'input_image': encoderInput},
      ['image_embeddings'],
    );
    encoderInput.release();
    encoderSession.release();

    final scaled = Offset(
        point.dx * (1024 / image.width), point.dy * (1024 / image.height));
    final coords = Float32List.fromList([scaled.dx, scaled.dy, 0.0, 0.0]);
    final labels = Float32List.fromList([1.0, -1.0]);

    final decoderData = await rootBundle.load('assets/decoder.onnx');
    final decoderSession = OrtSession.fromBuffer(
        decoderData.buffer.asUint8List(), OrtSessionOptions());

    final decoderInputs = {
      'image_embeddings': embeddings[0]!,
      'point_coords':
          OrtValueTensor.createTensorWithDataList(coords, [1, 2, 2]),
      'point_labels': OrtValueTensor.createTensorWithDataList(labels, [1, 2]),
      'mask_input': OrtValueTensor.createTensorWithDataList(
          Float32List(1 * 1 * 256 * 256), [1, 1, 256, 256]),
      'has_mask_input': OrtValueTensor.createTensorWithDataList(
          Float32List.fromList([0.0]), [1]),
      'orig_im_size': OrtValueTensor.createTensorWithDataList(
          Float32List.fromList([
            image.height.toDouble(),
            image.width.toDouble(),
          ]),
          [2]),
    };

    final maskOutput =
        decoderSession.run(OrtRunOptions(), decoderInputs, ['masks']);
    decoderSession.release();

    final rawMask = maskOutput[0]!.value as List;
    final binary = <int>[];
    for (final row in rawMask[0][0] as List) {
      for (final v in row as List) {
        binary.add((v as double) > 0 ? 0 : 255);
      }
    }

    final mask = img.Image.fromBytes(
      width: image.width,
      height: image.height,
      bytes: Uint8List.fromList(binary).buffer,
      numChannels: 1,
      format: img.Format.uint8,
    );

    setState(() {
      _segmentationMask = Uint8List.fromList(img.encodePng(mask));
      _maskImage = mask;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Inpainting")),
      body: _imageFile == null
          ? const Center(child: Text("Brak zdjęcia"))
          : GestureDetector(
              onTapDown: (details) {
                final box =
                    imageKey.currentContext!.findRenderObject() as RenderBox;
                final local = box.globalToLocal(details.globalPosition);
                final boxSize = box.size;

                final scaleX = _imageWidth! / boxSize.width;
                final scaleY = _imageHeight! / boxSize.height;

                final imagePoint = Offset(local.dx * scaleX, local.dy * scaleY);
                _runSegmentationFromClick(imagePoint);
              },
              child: Center(
                child: SizedBox(
                  width: _imageWidth?.toDouble(),
                  height: _imageHeight?.toDouble(),
                  child: Image.file(
                    _imageFile!,
                    key: imageKey,
                    width: 1024,
                    height: 1024,
                    fit: BoxFit.fill,
                  ),
                ),
              ),
            ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
      floatingActionButton: Row(
        mainAxisSize: MainAxisSize.min,
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          FloatingActionButton(
            onPressed: _pickImage,
            heroTag: 'pick',
            child: const Icon(Icons.photo_library),
          ),
          const SizedBox(width: 16),
          FloatingActionButton(
            onPressed: _startInpaintingWithSnackBar,
            heroTag: 'inpaint',
            child: const Icon(Icons.auto_fix_high),
          ),
        ],
      ),
    );
  }
}
