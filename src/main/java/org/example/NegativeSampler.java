package org.example;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.*;
import java.nio.file.*;
import java.util.*;

import static org.example.Main.CURRENT_STAGE;

public class NegativeSampler {
    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    private static final Size PATCH_SIZE = new Size(80, 200);

    private static final int NEGATIVES_PER_IMAGE = 5;

    public static void main(String[] args) throws IOException {
        String imageDir = "src\\main\\resources\\train\\";
        String annotationDir = "src\\main\\resources\\out\\" + CURRENT_STAGE + "\\annotations\\";
        String negativeDir = "src\\main\\resources\\out\\" + CURRENT_STAGE + "\\negatives\\";

        Files.createDirectories(Paths.get(negativeDir));

        for (File imgFile : new File(imageDir).listFiles((d, n) -> n.endsWith(".png"))) {
            String baseName = imgFile.getName().replace(".png", "");
            File annFile = new File(annotationDir + "/" + baseName + ".txt");

            if (!annFile.exists()) continue;

            Mat img = Imgcodecs.imread(imgFile.getAbsolutePath());
            if (img.empty()) continue;

            List<Rect> annotatedRects = loadAnnotations(annFile.getAbsolutePath());
            Random rand = new Random();
            int generated = 0;


            while (generated < NEGATIVES_PER_IMAGE) {
                int x = rand.nextInt(img.cols() - (int) PATCH_SIZE.width);
                int y = rand.nextInt(img.rows() - (int) PATCH_SIZE.height);
                Rect candidate = new Rect(x, y, (int) PATCH_SIZE.width, (int) PATCH_SIZE.height);

                boolean overlaps = annotatedRects.stream().anyMatch(r -> getIoU(candidate, r) > 0.2);
                if (!overlaps) {
                    Mat roi = new Mat(img, candidate);
                    Imgcodecs.imwrite(negativeDir + "/" + baseName + "_n" + generated + ".png", roi);
                    generated++;
                }
            }
        }
    }

    private static List<Rect> loadAnnotations(String path) {
        List<Rect> rects = new ArrayList<>();
        File file = new File(path);
        if (!file.exists()) return rects;

        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.trim().split("\\s+");
                if (parts.length == 4) {
                    int x0 = Integer.parseInt(parts[0]);
                    int y0 = Integer.parseInt(parts[1]);
                    int x1 = Integer.parseInt(parts[2]);
                    int y1 = Integer.parseInt(parts[3]);
                    rects.add(new Rect(x0, y0, x1 - x0, y1 - y0));
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return rects;
    }

    private static double getIoU(Rect a, Rect b) {
        int x1 = Math.max(a.x, b.x);
        int y1 = Math.max(a.y, b.y);
        int x2 = Math.min(a.x + a.width, b.x + b.width);
        int y2 = Math.min(a.y + a.height, b.y + b.height);
        int interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        int unionArea = a.width * a.height + b.width * b.height - interArea;
        return unionArea == 0 ? 0 : (double) interArea / unionArea;
    }
}

