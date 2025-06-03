import 'dart:io';
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

      final resultBytes = Uint8List.fromList(img.encodePng(decoded));
      final tempFile = await File(
              '${(await Directory.systemTemp.createTemp()).path}/input.png')
          .writeAsBytes(resultBytes);

      FirebaseCrashlytics.instance.log("Obraz wybrany przez użytkownika");
      FirebaseCrashlytics.instance.setCustomKey(
          "image_resolution", "${decoded.width}x${decoded.height}");

      setState(() {
        _imageFile = tempFile;
        _imageWidth = decoded.width;
        _imageHeight = decoded.height;
        _segmentationMask = null;
        _maskImage = img.Image(
            width: decoded.width, height: decoded.height, numChannels: 1)
          ..getBytes().fillRange(0, decoded.width * decoded.height, 255);
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Inpainting")),
      body: _imageFile == null
          ? const Center(child: Text("Brak zdjęcia"))
          : Center(
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
