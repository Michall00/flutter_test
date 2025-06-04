import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';
import '../utils/tensor_utils.dart';

class InpaintingService {
  static Future<Uint8List> runInpainting({
    required img.Image original,
    required img.Image mask,
    required ByteData modelData,
  }) async {
    OrtEnv.instance.init();
    final session = OrtSession.fromBuffer(
      modelData.buffer.asUint8List(),
      OrtSessionOptions(),
    );

    final imageTensor = convertImageToUint8NCHW(original);
    final maskTensor = convertMaskToUint8NCHW(mask);

    final result = session.run(
      OrtRunOptions(),
      {'image': imageTensor, 'mask': maskTensor},
      ['result'],
    );

    imageTensor.release();
    maskTensor.release();
    session.release();

    final output = result[0]!.value as List;
    final imgOut = convertNCHWtoImage(output);
    return Uint8List.fromList(img.encodeJpg(imgOut));
  }
}
