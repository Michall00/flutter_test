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

    final maskImage = img.Image(width: width, height: height);

    final white = img.ColorRgb8(255, 255, 255);
    final black = img.ColorRgb8(0, 0, 0);

    for (int y = 0; y < maskImage.height; y++) {
      for (int x = 0; x < maskImage.width; x++) {
        maskImage.setPixel(x, y, white);
      }
    }

    for (final point in _points) {
      // img.drawCircle(maskImage, point.dx.toInt(), point.dy.toInt(), 15, img.getColor(0, 0, 0), thickness: -1);
      img.drawCircle(
      maskImage,
      // center: Point(point.dx.toInt(), point.dy.toInt()),
      x: point.dx.toInt(),
      y: point.dy.toInt(),
      radius: 15,
      color: black
    );
    }

    final inputImage = img.copyRotate(originalImage, angle: 0);
    final imageBytes = inputImage.getBytes(order: img.ChannelOrder.rgb);
    final maskBytes = Uint8List.fromList(
      maskImage
          .getBytes()
          .asMap()
          .entries
          .where((entry) => entry.key % 4 == 0)
          .map((entry) => entry.value)
          .toList(),
    );

    final modelData = await rootBundle.load('assets/migan_pipeline_v2.onnx');
    final session = OrtSession.fromBuffer(
      modelData.buffer.asUint8List(),
      OrtSessionOptions(),
    );

    final imageTensor = OrtValueTensor.createTensorWithDataList(imageBytes, [1, height, width, 3]);
    final maskTensor = OrtValueTensor.createTensorWithDataList(maskBytes, [1, height, width, 1]);


    final options = OrtRunOptions();
    final outputs = await session.run(
      options,
      {'image': imageTensor, 'mask': maskTensor},
      ['result'],
    );
    final output = outputs.first?.value as Uint8List;
    final outputImage = img.Image.fromBytes(
      width: width,
      height: height,
      bytes: output.buffer,
      order: img.ChannelOrder.rgb,
    );

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
