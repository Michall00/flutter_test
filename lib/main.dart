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

  runZonedGuarded(
    () => runApp(const MyApp()),
    (error, stackTrace) =>
        FirebaseCrashlytics.instance.recordError(error, stackTrace),
  );
}

final GlobalKey imageKey = GlobalKey();

enum InteractionMode { draw, segment }

InteractionMode _mode = InteractionMode.segment;

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
  Uint8List? _previewMaskBytes;
  final List<Offset> _points = [];
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
      const targetSize = 512;

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
          "image_resolution", "${decoded.width}x${decoded.height}");

      setState(() {
        _imageFile = tempFile;
        _imageWidth = targetSize;
        _imageHeight = targetSize;
        _points.clear();
        _segmentationMask = null;
        _maskImage =
            img.Image(width: targetSize, height: targetSize, numChannels: 1)
              ..getBytes().fillRange(0, targetSize * targetSize, 255);
      });
    }
  }

  Future<void> _runSegmentationFromClick(Offset point) async {
    FirebaseCrashlytics.instance.log(
        "🎯 Start segmentacji na pozycji: ${point.dx.toStringAsFixed(1)}, ${point.dy.toStringAsFixed(1)}");
    FirebaseCrashlytics.instance.setCustomKey("segment_point_x", point.dx);
    FirebaseCrashlytics.instance.setCustomKey("segment_point_y", point.dy);
    final messenger = ScaffoldMessenger.of(context);
    messenger.showSnackBar(
      const SnackBar(
        content: Text('Uruchomienie segmentacji...'),
        duration: Duration(seconds: 1),
      ),
    );
    if (_imageFile == null) return;

    final bytes = await _imageFile!.readAsBytes();
    final image = img.decodeImage(bytes)!;

    FirebaseCrashlytics.instance
        .setCustomKey("image_dims", "${image.width}x${image.height}");

    messenger.showSnackBar(
      const SnackBar(
        content: Text('Wczytenie pliku do segmentacji...'),
        duration: Duration(seconds: 1),
      ),
    );

    final pixels = image.getBytes(order: img.ChannelOrder.rgb);
    final imgFloat = Float32List(image.width * image.height * 3);
    for (int i = 0; i < pixels.length; i++) {
      imgFloat[i] = pixels[i].toDouble();
    }

    final encoderInput = OrtValueTensor.createTensorWithDataList(
        imgFloat, [image.height, image.width, 3]);
    messenger.showSnackBar(
      const SnackBar(
        content: Text('Załadowanie sesji...'),
        duration: Duration(seconds: 1),
      ),
    );
    final encoderData = await rootBundle.load('assets/encoder.onnx');
    final encoderSession = OrtSession.fromBuffer(
        encoderData.buffer.asUint8List(), OrtSessionOptions());
    messenger.showSnackBar(
      const SnackBar(
        content: Text('Uruchomienie enkodera...'),
        duration: Duration(seconds: 1),
      ),
    );
    final embeddings = encoderSession.run(
      OrtRunOptions(),
      {'input_image': encoderInput},
      ['image_embeddings'],
    );
    encoderInput.release();
    encoderSession.release();
    messenger.showSnackBar(
      const SnackBar(
        content:
            Text('Output z enkodera i przygotowanie danych do dekodera...'),
        duration: Duration(seconds: 1),
      ),
    );
    FirebaseCrashlytics.instance.log("🧠 Zakończono inferencję enkodera");

    final scaled = Offset(
        point.dx * (1024 / image.width), point.dy * (1024 / image.height));
    final coords = Float32List.fromList([scaled.dx, scaled.dy, 0.0, 0.0]);
    final labels = Float32List.fromList([1.0, -1.0]);

    messenger.showSnackBar(
      const SnackBar(
        content: Text('Uruchomienie dekodera...'),
        duration: Duration(seconds: 1),
      ),
    );

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
          Float32List.fromList(
              [image.height.toDouble(), image.width.toDouble()]),
          [2]),
    };

    final maskOutput =
        decoderSession.run(OrtRunOptions(), decoderInputs, ['masks']);

    decoderSession.release();
    messenger.showSnackBar(
      const SnackBar(
        content: Text('Koniec dekodera...'),
        duration: Duration(seconds: 1),
      ),
    );
    FirebaseCrashlytics.instance.log("🧠 Zakończono inferencję dekodera");
    FirebaseCrashlytics.instance.setCustomKey("mask_output_size", "512x512");

    final rawMask = maskOutput[0]!.value as List;
    final binary = <int>[];
    for (final row in rawMask[0][0] as List) {
      for (final v in row as List) {
        binary.add((v as double) > 0 ? 0 : 255);
      }
    }

    final mask = img.Image.fromBytes(
      width: 512,
      height: 512,
      bytes: Uint8List.fromList(binary).buffer,
      numChannels: 1,
      format: img.Format.uint8,
    );

    messenger.showSnackBar(
      const SnackBar(
        content: Text('Przetworzenie maski po dekoderze...'),
        duration: Duration(seconds: 1),
      ),
    );
    setState(() {
      _segmentationMask = Uint8List.fromList(img.encodePng(mask));
      _maskImage = mask;
    });
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

    messenger.showSnackBar(
      const SnackBar(
        content: Text('Wczytenie pliku do inpaintingu...'),
        duration: Duration(seconds: 1),
      ),
    );

    // const targetSize = 512;
    final bytes = await _imageFile!.readAsBytes();
    final originalImage = img.decodeImage(bytes)!;
    // final width = originalImage.width;
    // final height = originalImage.height;

    // final maskImage = img.Image(width: width, height: height, numChannels: 1);

    // maskImage.getBytes().fillRange(0, width * height, 255);
    final pointsToDraw = _points.where((p) => p != Offset.infinite).toList();
    messenger.showSnackBar(
      SnackBar(
        content: Text('Liczba punktów: ${pointsToDraw.length}'),
        duration: const Duration(seconds: 1),
      ),
    );

    if (_segmentationMask != null) {
      _maskImage = img.decodeImage(_segmentationMask!)!;
    }

    // final resizedImage = img.copyResize(
    //   originalImage,
    //   width: targetSize,
    //   height: targetSize,
    //   interpolation: img.Interpolation.linear,
    // );

    // final resizedMask = img.copyResize(
    //   _maskImage!,
    //   width: targetSize,
    //   height: targetSize,
    //   interpolation: img.Interpolation.nearest,
    // );

    messenger.showSnackBar(
      const SnackBar(
        content: Text('Generowanie tensorów wejściowych...'),
        duration: Duration(seconds: 1),
      ),
    );
    final imageTensor = convertImageToUint8NCHW(originalImage);
    final maskTensor = convertMaskToUint8NCHW(_maskImage!);

    OrtEnv.instance.init();
    final modelData = await rootBundle.load('assets/migan.onnx');
    final session = OrtSession.fromBuffer(
      modelData.buffer.asUint8List(),
      OrtSessionOptions(),
    );

    messenger.showSnackBar(
      const SnackBar(
        content: Text('Uruchomienie ONNX.'),
        duration: Duration(seconds: 1),
      ),
    );

    // final inputs = {'image': imageTensor, 'mask': maskTensor};
    // final outputNames = ['result'];

    // final options = OrtRunOptions();
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
    messenger.showSnackBar(
      const SnackBar(
        content: Text('Wynik ONNX uzyskany.'),
        duration: Duration(seconds: 1),
      ),
    );
    final output = result[0]!.value as List;
    final imgOut = convertNCHWtoImage(output);
    FirebaseCrashlytics.instance.log("✅ MI-GAN inference zakończony sukcesem");
    FirebaseCrashlytics.instance
        .setCustomKey("result_size", "${output.length} px");

    messenger.showSnackBar(
      SnackBar(
        content: Text("Wynik ONNX to Uint8List (${output.length} bajtów)"),
        duration: const Duration(seconds: 1),
      ),
    );

    final upscaledResult = img.copyResize(
      imgOut,
      width: _imageWidth!,
      height: _imageHeight!,
      interpolation: img.Interpolation.linear,
    );

    final resultBytes = Uint8List.fromList(img.encodeJpg(upscaledResult));
    debugPrint("📦 Zakodowano wynik do JPG (${resultBytes.length} bajtów)");

    final tempDir = await Directory.systemTemp.createTemp();
    final filePath = '${tempDir.path}/output.jpg';
    final resultFile = await File(filePath).writeAsBytes(resultBytes);

    setState(() {
      _imageFile = resultFile;
      _points.clear();
      _previewMaskBytes = null;
    });
  }

  void _generateMaskPreview() {
    if (_imageFile == null || _imageWidth == null || _imageHeight == null) {
      return;
    }

    final original = img.decodeImage(_imageFile!.readAsBytesSync())!;
    final preview = img.Image.from(original);

    final mask =
        img.Image(width: _imageWidth!, height: _imageHeight!, numChannels: 1);
    mask.getBytes().fillRange(0, _imageWidth! * _imageHeight!, 255);

    for (final point in _points) {
      if (point == Offset.infinite) continue;

      final x = point.dx.toInt().clamp(0, _imageWidth! - 1);
      final y = point.dy.toInt().clamp(0, _imageHeight! - 1);
      img.drawCircle(mask, x: x, y: y, radius: 15, color: img.ColorUint8(0));
    }

    for (int y = 0; y < mask.height; y++) {
      for (int x = 0; x < mask.width; x++) {
        final pixel = mask.getPixel(x, y);
        final value = pixel.getChannel(img.Channel.luminance);
        if (value == 0) {
          preview.setPixelRgba(x, y, 255, 0, 0, 100);
        }
      }
    }

    final bytes = Uint8List.fromList(img.encodeJpg(preview));
    setState(() {
      _previewMaskBytes = bytes;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("MI-GAN Inpainting")),
      body: _imageFile == null
          ? const Center(child: Text("Brak zdjęcia"))
          : Center(
              child: SizedBox(
                width: _imageWidth?.toDouble(),
                height: _imageHeight?.toDouble(),
                child: Stack(
                  children: [
                    _previewMaskBytes != null
                        ? Image.memory(_previewMaskBytes!)
                        : Image.file(
                            _imageFile!,
                            key: imageKey,
                            width: _imageWidth!.toDouble(),
                            height: _imageHeight!.toDouble(),
                          ),
                    GestureDetector(
                      onTapDown: (details) {
                        if (_mode == InteractionMode.segment) {
                          final box = imageKey.currentContext!
                              .findRenderObject() as RenderBox;
                          final local =
                              box.globalToLocal(details.globalPosition);
                          _runSegmentationFromClick(local);
                        }
                      },
                      onPanUpdate: (details) {
                        if (_mode == InteractionMode.draw) {
                          setState(() => _points.add(details.localPosition));
                        }
                      },
                      onPanEnd: (_) {
                        if (_mode == InteractionMode.draw) {
                          _points.add(Offset.infinite);
                        }
                      },
                      child: CustomPaint(
                        painter: MaskPainter(_points),
                        size: Size(
                            _imageWidth!.toDouble(), _imageHeight!.toDouble()),
                      ),
                    ),
                    Positioned(
                      top: 16,
                      right: 16,
                      child: FloatingActionButton(
                        onPressed: () {
                          setState(() {
                            _mode = _mode == InteractionMode.draw
                                ? InteractionMode.segment
                                : InteractionMode.draw;
                          });
                        },
                        child: Icon(
                          _mode == InteractionMode.draw
                              ? Icons.edit
                              : Icons.crop_free,
                        ),
                      ),
                    ),
                  ],
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
          const SizedBox(width: 16),
          FloatingActionButton(
            onPressed: _generateMaskPreview,
            heroTag: 'preview',
            child: const Icon(Icons.visibility),
          ),
          const SizedBox(width: 16),
          FloatingActionButton(
            onPressed: () {
              setState(() {
                _points.clear();
                _previewMaskBytes = null;
              });
            },
            heroTag: 'clear',
            child: const Icon(Icons.clear),
          ),
          FloatingActionButton(
            onPressed: () {
              setState(() {
                _mode = _mode == InteractionMode.draw
                    ? InteractionMode.segment
                    : InteractionMode.draw;
              });
            },
            heroTag: 'mode',
            child: Icon(
              _mode == InteractionMode.draw ? Icons.brush : Icons.touch_app,
            ),
          ),
        ],
      ),
    );
  }
}

class MaskPainter extends CustomPainter {
  final List<Offset> points;
  MaskPainter(this.points);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.red.withOpacity(0.5)
      ..strokeCap = StrokeCap.round
      ..strokeWidth = 20;

    for (int i = 0; i < points.length - 1; i++) {
      if (points[i] != Offset.infinite && points[i + 1] != Offset.infinite) {
        canvas.drawLine(points[i], points[i + 1], paint);
      }
    }
  }

  @override
  bool shouldRepaint(MaskPainter oldDelegate) => true;
}
