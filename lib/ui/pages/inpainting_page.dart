import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import '../widgets/mask_painter.dart';
import '../../services/image_service.dart';
import '../../services/inpainting_service.dart';
import '../../services/segmentation_service.dart';

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
    final imageBytes = await _imageFile!.readAsBytes();
    final decoded = img.decodeImage(imageBytes)!;
    final modelData = await rootBundle.load('assets/migan.onnx');

    final output = await InpaintingService.runInpainting(
      original: decoded,
      mask: _maskImage!,
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
              child: Stack(
                children: [
                  if (_outputBytes != null)
                    Image.memory(_outputBytes!, key: _imageKey)
                  else if (_maskBytes != null)
                    Image.memory(_maskBytes!, key: _imageKey)
                  else
                    Image.file(_imageFile!, key: _imageKey),
                  GestureDetector(
                    onTapDown: (details) {
                      if (_mode == InteractionMode.segment) {
                        final box = _imageKey.currentContext!.findRenderObject()
                            as RenderBox;
                        final local = box.globalToLocal(details.globalPosition);
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
                      size: Size.infinite,
                    ),
                  )
                ],
              ),
            ),
      floatingActionButton: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          FloatingActionButton(
              onPressed: _pickImage,
              heroTag: 'pick',
              child: const Icon(Icons.photo)),
          const SizedBox(width: 16),
          FloatingActionButton(
              onPressed: _runInpainting,
              heroTag: 'inpaint',
              child: const Icon(Icons.brush)),
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
          )
        ],
      ),
    );
  }
}

enum InteractionMode { draw, segment }
