import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:onnxruntime/onnxruntime.dart';

class ImagePickerPage extends StatefulWidget {
  @override
  State<ImagePickerPage> createState() => _ImagePickerPageState();
}

class _ImagePickerPageState extends State<ImagePickerPage> {
  File? _imageFile;
  final List<Offset> _points = [];

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
    if (_imageFile == null) return;

    final bytes = await _imageFile!.readAsBytes();
    final originalImage = img.decodeImage(bytes)!;
    final width = originalImage.width;
    final height = originalImage.height;

    final maskImage = img.Image(width, height);
    img.fill(maskImage, img.getColor(255, 255, 255));
    for (final point in _points) {
      // img.drawCircle(maskImage, point.dx.toInt(), point.dy.toInt(), 15, img.getColor(0, 0, 0), thickness: -1);
      img.drawCircle(
      maskImage,
      center: Point(point.dx.toInt(), point.dy.toInt()),
      radius: 15,
      color: img.getColor(0, 0, 0),
      thickness: -1,
    );
    }

    final inputImage = img.copyRotate(originalImage, 0);
    final imageBytes = inputImage.getBytes(format: img.Format.rgb);
    final maskBytes = maskImage.getBytes(format: img.Format.luminance);

    final imageTensor = Tensor.fromList(
      imageBytes,
      shape: [1, 3, height, width],
      elementType: TensorElementType.uint8,
    );

    final maskTensor = Tensor.fromList(
      maskBytes,
      shape: [1, 1, height, width],
      elementType: TensorElementType.uint8,
    );

    final modelData = await rootBundle.load('assets/migan_pipeline_v2.onnx');
    final session = OrtSession.fromBuffer(modelData.buffer.asUint8List());

    final outputs = await session.run({'image': imageTensor, 'mask': maskTensor});
    final output = outputs.first.value as List<int>;
    final outputImage = img.Image.fromBytes(width, height, Uint8List.fromList(output), format: img.Format.rgb);

    final resultBytes = Uint8List.fromList(img.encodeJpg(outputImage));
    setState(() {
      _imageFile = File.fromRawPath(resultBytes);
      _points.clear();
    });
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
            onPressed: _runInpainting,
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
