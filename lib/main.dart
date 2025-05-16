import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:onnxruntime/onnxruntime.dart';

void main() {
  runApp(const MyApp());
}

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
  int? _imageWidth;
  int? _imageHeight;
  final List<Offset> _points = [];

  void _startInpaintingWithSnackBar() async {
    final messenger = ScaffoldMessenger.of(context);

    try {
      await _runInpainting();

      messenger.showSnackBar(
        const SnackBar(
          content: Text('✅ Inpainting zakończony!'),
          duration: Duration(seconds: 2),
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

      setState(() {
        _imageFile = file;
        _imageWidth = decoded.width;
        _imageHeight = decoded.height;
        _points.clear();
      });
    }
  }

  Future<void> _runInpainting() async {
    final messenger = ScaffoldMessenger.of(context);
    if (_imageFile == null) {
      messenger.showSnackBar(
        const SnackBar(
          content: Text('Brak pliku obrazu!'),
          duration: Duration(seconds: 2),
        ),
      );
      return;
    }

    messenger.showSnackBar(
      const SnackBar(
        content: Text('Wczytenie pliku do inpaintingu...'),
        duration: Duration(seconds: 2),
      ),
    );
    final bytes = await _imageFile!.readAsBytes();
    final originalImage = img.decodeImage(bytes)!;
    final width = originalImage.width;
    final height = originalImage.height;

    final maskImage = img.Image(width: width, height: height, numChannels: 1);

    maskImage.getBytes().fillRange(0, width * height, 255);
    final pointsToDraw = _points.where((p) => p != Offset.infinite).toList();
    messenger.showSnackBar(
      SnackBar(
        content: Text('Liczba punktów: ${pointsToDraw.length}'),
        duration: const Duration(seconds: 1),
      ),
    );

    for (final point in _points) {
      if (point == Offset.infinite) continue;

      img.drawCircle(
        maskImage,
        x: point.dx.toInt(),
        y: point.dy.toInt(),
        radius: 15,
        color: img.ColorUint8(0),
      );
    }

    messenger.showSnackBar(
      const SnackBar(
        content: Text('Generowanie tensorów wejściowych...'),
        duration: Duration(seconds: 1),
      ),
    );
    final imageTensor = convertImageToUint8NCHW(originalImage);
    final maskTensor = convertMaskToUint8NCHW(maskImage);

    OrtEnv.instance.init();
    final modelData = await rootBundle.load('assets/migan_pipeline_v2.onnx');
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

    final inputs = {'image': imageTensor, 'mask': maskTensor};
    final outputNames = ['result'];

    final options = OrtRunOptions();
    final result = session.run(
      options,
      inputs,
      outputNames,
    );

    messenger.showSnackBar(
      const SnackBar(
        content: Text('Wynik ONNX uzyskany.'),
        duration: Duration(seconds: 1),
      ),
    );
    final output = result[0]!.value as List;
    final imgOut = convertNCHWtoImage(output);

    messenger.showSnackBar(
      SnackBar(
        content: Text("Wynik ONNX to Uint8List (${output.length} bajtów)"),
        duration: const Duration(seconds: 2),
      ),
    );

    final resultBytes = Uint8List.fromList(img.encodeJpg(imgOut));
    debugPrint("📦 Zakodowano wynik do JPG (${resultBytes.length} bajtów)");

    final tempDir = await Directory.systemTemp.createTemp();
    final filePath = '${tempDir.path}/output.jpg';
    final resultFile = await File(filePath).writeAsBytes(resultBytes);

    setState(() {
      _imageFile = resultFile;
      _points.clear();
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
                    Image.file(_imageFile!),
                    GestureDetector(
                      onPanUpdate: (details) {
                        setState(() => _points.add(details.localPosition));
                      },
                      onPanEnd: (_) => _points.add(Offset.infinite),
                      child: CustomPaint(
                        painter: MaskPainter(_points),
                        size: Size(
                            _imageWidth!.toDouble(), _imageHeight!.toDouble()),
                      ),
                    ),
                  ],
                ),
              ),
            ),
      floatingActionButton: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          FloatingActionButton(
            onPressed: _pickImage,
            heroTag: 'pick',
            child: const Icon(Icons.photo_library),
          ),
          const SizedBox(height: 16),
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

OrtValueTensor convertImageToUint8NCHW(img.Image image) {
  final w = image.width, h = image.height;
  final pixels = image.getBytes(order: img.ChannelOrder.rgb);

  final uint8s = <int>[];

  for (int c = 0; c < 3; c++) {
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final i = (y * w + x) * 3 + c;
        final value = pixels[i];
        uint8s.add(value);
      }
    }
  }

  return OrtValueTensor.createTensorWithDataList(Uint8List.fromList(uint8s), [
    1,
    3,
    h,
    w,
  ]);
}

OrtValueTensor convertMaskToUint8NCHW(img.Image mask) {
  final w = mask.width, h = mask.height;
  final pixels = mask.getBytes();

  final uint8s = <int>[];

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      final i = y * w + x;
      final value = pixels[i];
      uint8s.add(value);
    }
  }

  return OrtValueTensor.createTensorWithDataList(Uint8List.fromList(uint8s), [
    1,
    1,
    h,
    w,
  ]);
}

img.Image convertNCHWtoImage(List data) {
  final channels = data[0];
  final h = channels[0].length;
  final w = channels[0][0].length;
  final image = img.Image(width: w, height: h);

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      final r = channels[0][y][x];
      final g = channels[1][y][x];
      final b = channels[2][y][x];
      image.setPixelRgb(x, y, r, g, b);
    }
  }

  return image;
}
