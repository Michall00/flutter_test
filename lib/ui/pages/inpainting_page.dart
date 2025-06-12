import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import '../widgets/mask_painter.dart';
import '../../services/image_service.dart';
import '../../services/inpainting_service.dart';
import '../../services/segmentation_service.dart';
import '../../utils/image_utils.dart';
import 'package:firebase_crashlytics/firebase_crashlytics.dart';
import 'package:image_gallery_saver/image_gallery_saver.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:firebase_analytics/firebase_analytics.dart';

class InpaintingPage extends StatefulWidget {
  const InpaintingPage({super.key});

  @override
  State<InpaintingPage> createState() => _InpaintingPageState();
}

class _InpaintingPageState extends State<InpaintingPage> {
  File? _imageFile;
  img.Image? _maskImage;
  int? _imageWidth;
  int? _imageHeight;
  Uint8List? _previewMaskBytes;
  final List<Offset> _points = [];
  Uint8List? _segmentationMask;
  final GlobalKey _imageKey = GlobalKey();
  Uint8List? _outputBytes;
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
      _imageWidth = resized.width;
      _imageHeight = resized.height;
      _points.clear();
      _segmentationMask = null;
      _previewMaskBytes = null;
      _maskImage = img.Image(
          width: resized.width, height: resized.height, numChannels: 1)
        ..getBytes().fillRange(0, resized.width * resized.height, 255);
    });
    FirebaseAnalytics.instance.logEvent(
      name: 'image_picked',
      parameters: {
        'width': _imageWidth!,
        'height': _imageHeight!,
        'source': 'gallery',
      },
    );
  }

  Future<void> _saveImageToGallery(Uint8List imageBytes) async {
    try {
      final status = await Permission.photosAddOnly.request();
      if (!status.isGranted) {
        FirebaseCrashlytics.instance.log("Gallery permission denied");
        return;
      }

      final result = await ImageGallerySaver.saveImage(
        imageBytes,
        quality: 100,
        name: "inpainted_${DateTime.now().millisecondsSinceEpoch}.png",
      );
      if (result['isSuccess']) {
        FirebaseCrashlytics.instance.log("Image saved to gallery");
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Zapisano obraz do galerii')),
        );
      } else {
        FirebaseCrashlytics.instance.log("Image not saved");
      }
    } catch (e, s) {
      FirebaseCrashlytics.instance.recordError(e, s);
    }
  }

  Future<void> _pickImageFromCamera() async {
    try {
      final picker = ImagePicker();
      final pickedFile = await picker.pickImage(source: ImageSource.camera);

      if (pickedFile == null) return;

      final file = File(pickedFile.path);

      setState(() {
        _imageFile = file;
        _maskImage = null;
        _previewMaskBytes = null;
        _points.clear();
      });

      FirebaseCrashlytics.instance.log("Image picked from camera");
    } catch (e, s) {
      FirebaseCrashlytics.instance.recordError(e, s);
    }
  }

  Future<void> _runSegmentationFromClick(Offset point) async {
    if (_imageFile == null) return;
    final encoderData = await rootBundle.load('assets/encoder.onnx');
    final decoderData = await rootBundle.load('assets/decoder.onnx');

    final segmentationStart = DateTime.now();
    FirebaseAnalytics.instance.logEvent(
      name: 'segmentation_started',
      parameters: {
        'x': point.dx,
        'y': point.dy,
      },
    );

    final mask = await SegmentationService.runSegmentation(
      imageFile: _imageFile!,
      clickPoint: point,
      encoderData: encoderData,
      decoderData: decoderData,
    );

    final segmentationEnd = DateTime.now();
    final durationMs =
        segmentationEnd.difference(segmentationStart).inMilliseconds;

    FirebaseAnalytics.instance.logEvent(
      name: 'segmentation_completed',
      parameters: {
        'duration_ms': durationMs,
      },
    );

    final imageBytes = await _imageFile!.readAsBytes();
    final baseImage = img.decodeImage(imageBytes)!;
    final decodedMask = img.decodeImage(mask)!;

    final overlay = img.Image.from(baseImage);
    for (int y = 0; y < overlay.height; y++) {
      for (int x = 0; x < overlay.width; x++) {
        final v = decodedMask.getPixel(x, y).r;
        if (v == 0) {
          overlay.setPixelRgba(x, y, 255, 0, 0, 100);
        }
      }
    }

    setState(() {
      _segmentationMask = mask;
      _maskImage = decodedMask;
      _previewMaskBytes = Uint8List.fromList(img.encodePng(overlay));
    });
  }

  Future<void> _runInpainting() async {
    if (_imageFile == null || _maskImage == null) return;

    final inpaintingStart = DateTime.now();
    FirebaseAnalytics.instance.logEvent(
      name: 'inpainting_started',
      parameters: {
        'width': _imageWidth!,
        'height': _imageHeight!,
      },
    );

    final bytes = await _imageFile!.readAsBytes();
    final originalImage = img.decodeImage(bytes)!;

    if (_segmentationMask != null) {
      _maskImage = img.decodeImage(_segmentationMask!)!;
    }

    final modelData = await rootBundle.load('assets/migan.onnx');

    final dilated = dilateMask(_maskImage!, radius: 20);

    final output = await InpaintingService.runInpainting(
      original: originalImage,
      mask: dilated,
      modelData: modelData,
    );

    final inpaintingEnd = DateTime.now();
    final durationMs = inpaintingEnd.difference(inpaintingStart).inMilliseconds;

    FirebaseAnalytics.instance.logEvent(
      name: 'inpainting_completed',
      parameters: {
        'duration_ms': durationMs,
      },
    );

    setState(() {
      _previewMaskBytes = null;
      _points.clear();
      _imageFile = null;
      _outputBytes = output;
    });
  }

  Widget _buildImageStack() {
    final width = _imageWidth?.toDouble() ?? 256;
    final height = _imageHeight?.toDouble() ?? 256;

    return Center(
      child: SizedBox(
        width: width,
        height: height,
        child: Stack(
          fit: StackFit.expand,
          children: [
            _previewMaskBytes != null
                ? Image.memory(
                    _previewMaskBytes!,
                    key: _imageKey,
                    fit: BoxFit.contain,
                  )
                : Image.file(
                    _imageFile!,
                    key: _imageKey,
                    fit: BoxFit.contain,
                  ),
            GestureDetector(
              onTapDown: (details) {
                if (_mode == InteractionMode.segment) {
                  final box =
                      _imageKey.currentContext!.findRenderObject() as RenderBox;
                  final local = box.globalToLocal(details.globalPosition);
                  final boxSize = box.size;
                  final scaleX = _imageWidth! / boxSize.width;
                  final scaleY = _imageHeight! / boxSize.height;
                  final imagePoint =
                      Offset(local.dx * scaleX, local.dy * scaleY);
                  _runSegmentationFromClick(imagePoint);
                }
              },
              onPanUpdate: (details) {
                if (_mode == InteractionMode.draw) {
                  final local = details.localPosition;
                  final x = local.dx.toInt().clamp(0, _maskImage!.width - 1);
                  final y = local.dy.toInt().clamp(0, _maskImage!.height - 1);
                  _maskImage!.setPixelRgba(x, y, 0, 0, 0, 255);
                  setState(() => _points.add(local));
                }
              },
              onPanEnd: (_) {
                if (_mode == InteractionMode.draw) {
                  _points.add(Offset.infinite);
                }
              },
              child: CustomPaint(
                painter: MaskPainter(_points),
                size: Size(width, height),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildFloatingButtons() {
    return Row(
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
          onPressed: _pickImageFromCamera,
          heroTag: 'camera',
          tooltip: 'Zrób zdjęcie',
          child: const Icon(Icons.camera_alt),
        ),
        const SizedBox(width: 16),
        FloatingActionButton(
          onPressed: _runInpainting,
          heroTag: 'inpaint',
          child: const Icon(Icons.auto_fix_high),
        ),
        const SizedBox(width: 16),
        FloatingActionButton(
          onPressed: _outputBytes == null
              ? null
              : () => _saveImageToGallery(_outputBytes!),
          heroTag: 'save',
          tooltip: 'Zapisz do galerii',
          child: const Icon(Icons.save_alt),
        ),
        const SizedBox(width: 16),
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
              _mode == InteractionMode.draw ? Icons.brush : Icons.touch_app),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Inpainting")),
      body: Builder(
        builder: (_) {
          if (_outputBytes != null) {
            return Center(
              child: Image.memory(
                _outputBytes!,
                width: _imageWidth?.toDouble(),
                height: _imageHeight?.toDouble(),
                fit: BoxFit.contain,
              ),
            );
          }
          if (_imageFile == null) {
            return const Center(child: Text("Brak zdjęcia"));
          }
          return _buildImageStack();
        },
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
      floatingActionButton: _buildFloatingButtons(),
    );
  }
}

enum InteractionMode { draw, segment }
