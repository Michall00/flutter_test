import 'dart:typed_data';
import 'dart:io';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

class SegmentationService {
  static Future<Uint8List> runSegmentation({
    required File imageFile,
    required Offset clickPoint,
    required ByteData encoderData,
    required ByteData decoderData,
  }) async {
    final imageBytes = await imageFile.readAsBytes();
    final image = img.decodeImage(imageBytes)!;
    final pixels = image.getBytes(order: img.ChannelOrder.rgb);

    final imgFloat = Float32List(pixels.length);
    for (int i = 0; i < pixels.length; i++) {
      imgFloat[i] = pixels[i].toDouble();
    }

    final encoderSession = OrtSession.fromBuffer(
        encoderData.buffer.asUint8List(), OrtSessionOptions());

    final encoderInput = OrtValueTensor.createTensorWithDataList(
        imgFloat, [image.height, image.width, 3]);

    final embeddings = encoderSession.run(
        OrtRunOptions(), {'input_image': encoderInput}, ['image_embeddings']);

    encoderInput.release();
    encoderSession.release();

    final coords = Float32List.fromList([
      clickPoint.dx * (1024 / image.width),
      clickPoint.dy * (1024 / image.height),
      0.0,
      0.0
    ]);
    final labels = Float32List.fromList([1.0, -1.0]);

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
    return Uint8List.fromList(img.encodePng(mask));
  }
}
