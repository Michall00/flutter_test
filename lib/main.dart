import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
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
  final List<Offset> _points = [];

  void _startInpaintingWithSnackBar() async {
    final messenger = ScaffoldMessenger.of(context);

    try {
      await _runInpainting();

      messenger.showSnackBar(
        SnackBar(
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
          duration: Duration(seconds: 3),
        ),
      );
    }
  }

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(source: ImageSource.gallery);
    if (picked != null) {
      setState(() {
        _imageFile = File(picked.path);
        _points.clear();
      });
    }
  }

  Future<void> _runInpainting() async {
    final messenger = ScaffoldMessenger.of(context);
    if (_imageFile == null) {
      messenger.showSnackBar(
        SnackBar(
          content: Text('Brak pliku obrazu!'),
          duration: Duration(seconds: 2),
        ),
      );
      return;
    }

    const int targetWidth = 256;
    const int targetHeight = 256;

    messenger.showSnackBar(
      SnackBar(
        content: Text('Wczytenie pliku do inpaintingu...'),
        duration: Duration(seconds: 2),
      ),
    );
    final bytes = await _imageFile!.readAsBytes();
    final originalImage = img.decodeImage(bytes)!;
    final width = originalImage.width;
    final height = originalImage.height;

    final maskImage = img.Image(width: targetWidth, height: targetHeight);

    final white = img.ColorRgb8(255, 255, 255);
    final black = img.ColorRgb8(0, 0, 0);

    for (int y = 0; y < maskImage.height; y++) {
      for (int x = 0; x < maskImage.width; x++) {
        maskImage.setPixel(x, y, white);
      }
    }

    final pointsToDraw = _points.where((p) => p != Offset.infinite).toList();
    messenger.showSnackBar(
      SnackBar(
        content: Text('Liczba punktów: ${pointsToDraw.length}'),
        duration: Duration(seconds: 1),
      ),
    );
    final scaledPoints = _points
      .where((p) => p != Offset.infinite)
      .map((p) => Offset(
            p.dx * targetWidth / width,
            p.dy * targetHeight / height,
          ))
      .toList();

    for (final point in scaledPoints) {
      img.drawCircle(
        maskImage,
        x: point.dx.toInt(),
        y: point.dy.toInt(),
        radius: 15,
        color: black,
      );
    }


    messenger.showSnackBar(
      SnackBar(
        content: Text('Generowanie tensorów wejściowych...'),
        duration: Duration(seconds: 1),
      ),
    );
    final inputImage = img.copyResize(originalImage, width: targetWidth, height: targetHeight);
    final interleavedBytes = inputImage.getBytes(order: img.ChannelOrder.rgb);
    
    final imageBytes = convertInterleavedToNCHW(interleavedBytes, targetWidth, targetHeight);
    final maskBytes = Uint8List.fromList(
      List.generate(targetWidth * targetHeight, (i) {
        final pixel = maskImage.getPixel(i % targetWidth, i ~/ targetHeight);
        final luminance = img.getLuminanceRgb(
          pixel.r.toInt(),
          pixel.g.toInt(),
          pixel.b.toInt(),
        );
        return luminance.toInt() == 0 ? 1 : 0; 
      }),
    );
    

    messenger.showSnackBar(
      SnackBar(
        content: Text('Inicjalizacja środowiska ONNX...\nimageBytes: ${imageBytes.length} (oczekiwane: ${width * height * 3})\nmaskBytes: ${maskBytes.length} (oczekiwane: ${width * height})'),
        duration: Duration(seconds: 3),
      ),
    );
    OrtEnv.instance.init();
    final modelData = await rootBundle.load('assets/migan_pipeline_v2.onnx');
    final session = OrtSession.fromBuffer(
      modelData.buffer.asUint8List(),
      OrtSessionOptions(),
    );

    final imageTensor = OrtValueTensor.createTensorWithDataList(imageBytes, [1, 3, targetHeight, targetWidth]);
    final maskTensor = OrtValueTensor.createTensorWithDataList(maskBytes, [1, 1, targetHeight, targetWidth]);

    messenger.showSnackBar(
      SnackBar(
        content: Text('Uruchomienie ONNX.'),
        duration: Duration(seconds: 1),
      ),
    );

    final options = OrtRunOptions();
    final outputs = await session.run(
      options,
      {'image': imageTensor, 'mask': maskTensor},
      ['result'],
    );

    messenger.showSnackBar(
      SnackBar(
        content: Text('Wynik ONNX uzyskany.'),
        duration: Duration(seconds: 1),
      ),
    );
    final output = outputs.first?.value as Uint8List;
    final outputImage = img.Image.fromBytes(
      width: width,
      height: height,
      bytes: output.buffer,
      order: img.ChannelOrder.rgb,
    );


    messenger.showSnackBar(
      SnackBar(
        content: Text('Kodowanie wyniku jako JPG...'),
        duration: Duration(seconds: 1),
      ),
    );
    final resultBytes = Uint8List.fromList(img.encodeJpg(outputImage));

    messenger.showSnackBar(
      SnackBar(
        content: Text('Aktualizacja obrazu w UI...'),
        duration: Duration(seconds: 1),
      ),
    );


    setState(() {
      _imageFile = File.fromRawPath(resultBytes);
      _points.clear();
    });

    messenger.showSnackBar(
      SnackBar(
        content: Text('Inpainting zakończony sukcesem!'),
        duration: Duration(seconds: 1),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("MI-GAN Inpainting")),
      body: _imageFile == null
          ? Center(child: Text("Brak zdjęcia"))
          : Stack(
              children: [
                Image.file(_imageFile!),
                GestureDetector(
                  onPanUpdate: (details) {
                    setState(() => _points.add(details.localPosition));
                  },
                  onPanEnd: (_) => _points.add(Offset.infinite),
                  child: CustomPaint(
                    painter: MaskPainter(_points),
                    size: Size.infinite,
                  ),
                ),
              ],
            ),
      floatingActionButton: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          FloatingActionButton(
            onPressed: _pickImage,
            heroTag: 'pick',
            child: Icon(Icons.photo_library),
          ),
          SizedBox(height: 16),
          FloatingActionButton(
            onPressed: _startInpaintingWithSnackBar,
            heroTag: 'inpaint',
            child: Icon(Icons.auto_fix_high),
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

Uint8List convertInterleavedToNCHW(Uint8List rgb, int width, int height) {
  final total = width * height;
  final r = Uint8List(total);
  final g = Uint8List(total);
  final b = Uint8List(total);

  for (int i = 0; i < total; i++) {
    r[i] = rgb[i * 3];
    g[i] = rgb[i * 3 + 1];
    b[i] = rgb[i * 3 + 2];
  }

  return Uint8List.fromList([...r, ...g, ...b]);
}