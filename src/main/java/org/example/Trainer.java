package org.example;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.SVM;
import org.opencv.ml.Ml;
import org.opencv.objdetect.HOGDescriptor;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import static org.example.Main.CURRENT_STAGE;

public class Trainer {
    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    private static final Size WINDOW_SIZE = new Size(80, 200);

    public static void main(String[] args) {
        List<Mat> features = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();

        HOGDescriptor hog = new HOGDescriptor(
                WINDOW_SIZE,
                new Size(16, 16),
                new Size(8, 8),
                new Size(8, 8),
                9
        );

        loadSamples("src\\main\\resources\\out\\" + CURRENT_STAGE + "\\positives\\", 1, hog, features, labels);
        loadSamples("src\\main\\resources\\out\\" + CURRENT_STAGE + "\\negatives\\", 0, hog, features, labels);

        Mat trainingData = new Mat(features.size(), features.get(0).cols(), CvType.CV_32F);
        for (int i = 0; i < features.size(); i++) {
            features.get(i).copyTo(trainingData.row(i));
        }

        Mat labelsMat = new Mat(labels.size(), 1, CvType.CV_32S);
        for (int i = 0; i < labels.size(); i++) {
            labelsMat.put(i, 0, labels.get(i));
        }

        SVM svm = SVM.create();
        svm.setKernel(SVM.LINEAR);
        svm.setType(SVM.C_SVC);
        svm.setC(0.01);

        svm.train(trainingData, Ml.ROW_SAMPLE, labelsMat);
        svm.save("model.yml");

        System.out.println("Training complete. model saved in 'model.yml'.");
    }

    private static void loadSamples(String folder, int label, HOGDescriptor hog, List<Mat> features, List<Integer> labels) {
        File dir = new File(folder);
        if (!dir.exists()) {
            System.err.println("Directory not found: " + folder);
            return;
        }

        for (File file : dir.listFiles((d, name) -> name.endsWith(".png"))) {
            Mat img = Imgcodecs.imread(file.getAbsolutePath(), Imgcodecs.IMREAD_GRAYSCALE);
            if (img.empty()) continue;

            Imgproc.resize(img, img, WINDOW_SIZE);

            MatOfFloat descriptors = new MatOfFloat();
            hog.compute(img, descriptors);

            features.add(descriptors.reshape(1, 1)); // 1 рядок
            labels.add(label);
        }
        System.out.println("Loaded " + features.size() + " samples from " + folder);
    }
}

