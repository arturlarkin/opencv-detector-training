package org.example;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.HOGDescriptor;
import org.opencv.ml.SVM;

import java.io.*;
import java.nio.file.*;

public class Main {
    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public static final String CURRENT_STAGE = "stage_3";

    private static final Size WINDOW_SIZE = new Size(80, 200);

    public static void main(String[] args) throws IOException {
        String inputDir = "src\\main\\resources\\test_public\\";
        String outputDir = "src\\main\\resources\\out\\" + CURRENT_STAGE + "\\img\\";
        String annotationFile = "src\\main\\resources\\out\\" + CURRENT_STAGE + "\\annotations.txt";

        AutoAnnotator.main(new String[]{});
        NegativeSampler.main(new String[]{});
        Trainer.main(new String[]{});

        Files.createDirectories(Paths.get(outputDir));
        BufferedWriter writer = new BufferedWriter(new FileWriter(annotationFile));

        HOGDescriptor hog = new HOGDescriptor(
                WINDOW_SIZE,
                new Size(16, 16),
                new Size(8, 8),
                new Size(8, 8),
                9
        );

        SVM svm = SVM.load("model.yml");
        MatOfFloat customDetector = getSvmDetectorVector(svm);
        hog.setSVMDetector(customDetector);

        try (DirectoryStream<Path> stream = Files.newDirectoryStream(Paths.get(inputDir), "*.png")) {
            for (Path entry : stream) {
                String filename = entry.getFileName().toString();
                String baseName = filename.replace(".png", "");

                Mat image = Imgcodecs.imread(entry.toString());
                if (image.empty()) continue;

                MatOfRect found = new MatOfRect();
                MatOfDouble weights = new MatOfDouble();
                hog.detectMultiScale(image, found, weights);

                for (Rect rect : found.toArray()) {
                    int x0 = rect.x;
                    int x1 = x0 + 80;
                    int y0 = 0;
                    int y1 = 200;

                    x1 = Math.min(x1, image.cols());
                    y1 = Math.min(y1, image.rows());

                    Imgproc.rectangle(image, new Point(x0, y0), new Point(x1, y1), new Scalar(0, 255, 0), 2);

                    writer.write(String.format("%s %d %d %d %d%n", baseName, y0, x0, y1, x1));
                }

                Imgcodecs.imwrite(outputDir + "/" + baseName + "_out.png", image);
            }
        }

        writer.close();
        System.out.println("Analysis complete.");
    }

    private static MatOfFloat getSvmDetectorVector(SVM svm) {
        Mat sv = svm.getSupportVectors();
        int svCount = sv.rows();
        int descriptorDim = sv.cols();

        Mat alpha = new Mat();
        Mat svidx = new Mat();
        double rho = svm.getDecisionFunction(0, alpha, svidx);

        Mat detector = new Mat(1, descriptorDim + 1, CvType.CV_32F);
        for (int i = 0; i < descriptorDim; i++) {
            float val = (float) (-sv.get(0, i)[0]);
            detector.put(0, i, val);
        }
        detector.put(0, descriptorDim, (float) rho);

        float[] detectorArray = new float[(int) detector.total()];
        detector.get(0, 0, detectorArray);
        return new MatOfFloat(detectorArray);
    }
}