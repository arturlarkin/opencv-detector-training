package org.example;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.objdetect.HOGDescriptor;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.SVM;

import java.io.*;
import java.nio.file.*;

import static org.example.Main.CURRENT_STAGE;

public class AutoAnnotator {
    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    private static final Size WINDOW_SIZE = new Size(80, 200);

    public static void main(String[] args) throws IOException {
        String imageDir = "src\\main\\resources\\train\\";
        String annotationDir = "src\\main\\resources\\out\\" + CURRENT_STAGE + "\\annotations\\";
        String positiveDir = "src\\main\\resources\\out\\" + CURRENT_STAGE + "\\positives\\";

        Files.createDirectories(Paths.get(annotationDir));
        Files.createDirectories(Paths.get(positiveDir));

        // HOG + SVM for version 1
        // HOGDescriptor hog = new HOGDescriptor();
        // hog.setSVMDetector(HOGDescriptor.getDefaultPeopleDetector());

        // HOG + SVM for subsequent versions

        HOGDescriptor hog = new HOGDescriptor(
                WINDOW_SIZE,
                new Size(16, 16),
                new Size(8, 8),
                new Size(8, 8),
                9
        );
        SVM svm = SVM.load("model.yml");
        hog.setSVMDetector(getSvmDetectorVector(svm));

        for (File imgFile : new File(imageDir).listFiles((d, n) -> n.endsWith(".png"))) {
            Mat img = Imgcodecs.imread(imgFile.getAbsolutePath());
            if (img.empty()) continue;

            MatOfRect detections = new MatOfRect();
            MatOfDouble weights = new MatOfDouble();

            hog.detectMultiScale(img, detections, weights, 0.0, new Size(8, 8), new Size(0, 0), 1.05, 1.0, false);

            String name = imgFile.getName().replace(".png", "");

            Rect[] rects = detections.toArray();

            double[] confs = new double[rects.length];
            if (weights.rows() > 0 && weights.cols() == 1) {
                confs = weights.toArray();
            }

            int idx = 0;
            StringBuilder annotationContent = new StringBuilder();

            for (int i = 0; i < rects.length; i++) {
                if (confs[i] < 0.4) continue;
                Rect r = clipRect(rects[i], img);

                annotationContent.append(String.format("%d %d %d %d%n", r.x, r.y, r.x + r.width, r.y + r.height));

                Mat roi = new Mat(img, r);
                Imgproc.resize(roi, roi, new Size(80, 200));
                Imgcodecs.imwrite(positiveDir + "/" + name + "_p" + idx++ + ".png", roi);
            }

            if (annotationContent.length() > 0) {
                File annFile = new File(annotationDir, name + ".txt");
                try (PrintWriter writer = new PrintWriter(annFile)) {
                    writer.print(annotationContent);
                }
            }
        }
    }

    private static Rect clipRect(Rect r, Mat img) {
        int x = Math.max(r.x, 0);
        int y = Math.max(r.y, 0);
        int x1 = Math.min(r.x + r.width, img.cols());
        int y1 = Math.min(r.y + r.height, img.rows());
        return new Rect(x, y, x1 - x, y1 - y);
    }

    private static MatOfFloat getSvmDetectorVector(SVM svm) {
        Mat sv = svm.getSupportVectors();
        Mat alpha = new Mat();
        Mat svidx = new Mat();
        double rho = svm.getDecisionFunction(0, alpha, svidx);

        Mat detector = new Mat(1, sv.cols() + 1, CvType.CV_32F);
        for (int i = 0; i < sv.cols(); i++) {
            detector.put(0, i, -sv.get(0, i)[0]);
        }
        detector.put(0, sv.cols(), (float) rho);

        float[] vec = new float[(int) detector.total()];
        detector.get(0, 0, vec);
        return new MatOfFloat(vec);
    }
}
