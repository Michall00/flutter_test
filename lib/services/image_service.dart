import 'dart:io';
import 'dart:typed_data';
import 'package:image/image.dart' as img;

class ImageService {
  static Future<img.Image?> decodeAndResize(File file, int targetSize) async {
    final bytes = await file.readAsBytes();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) return null;
    return img.copyResize(decoded,
        width: targetSize,
        height: targetSize,
        interpolation: img.Interpolation.linear);
  }

  static Future<File> saveTempImage(Uint8List bytes, String name) async {
    final tempDir = await Directory.systemTemp.createTemp();
    final filePath = '${tempDir.path}/$name';
    return File(filePath).writeAsBytes(bytes);
  }
}
