import 'package:firebase_crashlytics/firebase_crashlytics.dart';
import 'package:flutter/widgets.dart';

class CrashlyticsUtils {
  static void log(String message) {
    FirebaseCrashlytics.instance.log(message);
  }

  static void setKey(String key, dynamic value) {
    if (value is int) {
      FirebaseCrashlytics.instance.setCustomKey(key, value);
    } else if (value is double) {
      FirebaseCrashlytics.instance.setCustomKey(key, value);
    } else if (value is bool) {
      FirebaseCrashlytics.instance.setCustomKey(key, value);
    } else {
      FirebaseCrashlytics.instance.setCustomKey(key, value.toString());
    }
  }

  static void recordError(dynamic error, StackTrace stack) {
    FirebaseCrashlytics.instance.recordError(error, stack);
  }

  static void recordFlutterError(FlutterErrorDetails details) {
    FirebaseCrashlytics.instance.recordFlutterError(details);
  }
}
