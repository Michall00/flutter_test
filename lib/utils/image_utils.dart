import 'package:image/image.dart' as img;

img.Image dilateMask(img.Image mask, {int radius = 2}) {
  final result = img.Image.from(mask);
  final w = mask.width;
  final h = mask.height;

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      if (mask.getPixel(x, y).r == 255) {
        for (int dy = -radius; dy <= radius; dy++) {
          for (int dx = -radius; dx <= radius; dx++) {
            final nx = (x + dx).clamp(0, w - 1);
            final ny = (y + dy).clamp(0, h - 1);
            result.setPixelRgba(nx, ny, 255, 255, 255, 255);
          }
        }
      }
    }
  }

  return result;
}
