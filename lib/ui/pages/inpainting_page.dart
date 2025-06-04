import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import '../widgets/mask_painter.dart';
import '../../services/image_service.dart';
import '../../services/inpainting_service.dart';
import '../../services/segmentation_service.dart';
import 'package:image_picker/image_picker.dart';

class InpaintingPage extends StatefulWidget {
  const InpaintingPage({super.key});

  @override
  State<InpaintingPage> createState() => _InpaintingPageState();
}

class _InpaintingPageState extends State<InpaintingPage> {
  File? _imageFile;
  Uint8List? _maskBytes;
  Uint8List? _outputBytes;
  img.Image? _maskImage;
  final List<Offset> _points = [];
  final GlobalKey _imageKey = GlobalKey();
  InteractionMode _mode = InteractionMode.segment;

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(source: ImageSource.gallery);
    if (picked == null) return;
    final file = File(picked.path);
    final resized = await ImageService.decodeAndResize(file, 1024);
    if (resized == null) return;

    final resultBytes = Uint8List.fromList(img.encodePng(resized));
    final tempFile = await ImageService.saveTempImage(resultBytes, 'input.png');

    setState(() {
      _imageFile = tempFile;
      _outputBytes = null;
      _points.clear();
      _maskImage = img.Image(
          width: resized.width, height: resized.height, numChannels: 1)
        ..getBytes().fillRange(0, resized.width * resized.height, 255);
    });
  }

  Future<void> _runInpainting() async {
    if (_imageFile == null || _maskImage == null) return;

    final inverted = img.Image.from(_maskImage!);
    for (int y = 0; y < inverted.height; y++) {
      for (int x = 0; x < inverted.width; x++) {
        final pixel = inverted.getPixel(x, y);
        final v = pixel.r;
        final inv = 255 - v;
        inverted.setPixelRgba(x, y, inv, inv, inv, 255);
      }
    }

    final imageBytes = await _imageFile!.readAsBytes();
    final decoded = img.decodeImage(imageBytes)!;
    final modelData = await rootBundle.load('assets/migan.onnx');

    final output = await InpaintingService.runInpainting(
      original: decoded,
      mask: inverted,
      modelData: modelData,
    );

    setState(() => _outputBytes = output);
  }

  Future<void> _runSegmentation(Offset tap) async {
    if (_imageFile == null) return;
    final encoderData = await rootBundle.load('assets/encoder.onnx');
    final decoderData = await rootBundle.load('assets/decoder.onnx');
    final mask = await SegmentationService.runSegmentation(
      imageFile: _imageFile!,
      clickPoint: tap,
      encoderData: encoderData,
      decoderData: decoderData,
    );
    setState(() {
      _maskBytes = mask;
      _maskImage = img.decodeImage(mask);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Inpainting")),
      body: _imageFile == null
          ? const Center(child: Text("Brak zdjęcia"))
          : Center(
              child: FittedBox(
                fit: BoxFit.contain,
                child: SizedBox(
                  key: _imageKey,
                  width: 1024,
                  height: 1024,
                  child: Stack(
                    children: [
                      if (_outputBytes != null)
                        Image.memory(_outputBytes!)
                      else
                        Image.file(_imageFile!),
                      if (_maskBytes != null)
                        Image.memory(
                          _maskBytes!,
                          color: Colors.red.withOpacity(0.4),
                          colorBlendMode: BlendMode.srcATop,
                        ),
                      GestureDetector(
                        onTapDown: (details) {
                          if (_mode == InteractionMode.segment) {
                            final box = _imageKey.currentContext!
                                .findRenderObject() as RenderBox;
                            final local =
                                box.globalToLocal(details.globalPosition);
                            _runSegmentation(local);
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
                          size: const Size(1024, 1024),
                        ),
                      )
                    ],
                  ),
                ),
              ),
            ),
      floatingActionButton: Center(
        heightFactor: 1,
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            FloatingActionButton(
              onPressed: _pickImage,
              heroTag: 'pick',
              child: const Icon(Icons.photo),
            ),
            const SizedBox(width: 16),
            FloatingActionButton(
              onPressed: _runInpainting,
              heroTag: 'inpaint',
              child: const Icon(Icons.auto_fix_high),
            ),
            const SizedBox(width: 16),
            FloatingActionButton(
              onPressed: () {
                setState(() => _mode = _mode == InteractionMode.draw
                    ? InteractionMode.segment
                    : InteractionMode.draw);
              },
              heroTag: 'mode',
              child: Icon(
                  _mode == InteractionMode.draw ? Icons.edit : Icons.crop_free),
            ),
          ],
        ),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }
}

enum InteractionMode { draw, segment }
