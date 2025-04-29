#include <iostream>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
 
cv::Mat sequentialBlur(cv::Mat img) {
    // разбиваем изображение
    cv::Mat img_lt = img(cv::Rect(0, 0, 1000, 600));
    cv::Mat img_rt = img(cv::Rect(1000, 0, 1000, 600));
    cv::Mat img_lb = img(cv::Rect(0, 600, 1000, 600));
    cv::Mat img_rb = img(cv::Rect(1000, 600, 1000, 600));
 
    // применяем блюр
    cv::GaussianBlur(img_lt, img_lt, cv::Size(5, 5), 0);
    cv::GaussianBlur(img_rt, img_rt, cv::Size(5, 5), 0);
    cv::GaussianBlur(img_lb, img_lb, cv::Size(5, 5), 0);
    cv::GaussianBlur(img_rb, img_rb, cv::Size(5, 5), 0);
 
    // конкатенируем его части
    cv::hconcat(img_lt, img_rt, img_lt);
    cv::hconcat(img_lb, img_rb, img_lb);
    cv::vconcat(img_lt, img_lb, img);
    return img;
}

void blur_img(cv::Mat& img) {
    cv::GaussianBlur(img, img, cv::Size(5, 5), 0);
};
 
cv::Mat parallellBlurThreads(cv::Mat img) {

    cv::Mat img_lt = img(cv::Rect(0, 0, 1000, 600));
    cv::Mat img_rt = img(cv::Rect(1000, 0, 1000, 600));
    cv::Mat img_lb = img(cv::Rect(0, 600, 1000, 600));
    cv::Mat img_rb = img(cv::Rect(1000, 600, 1000, 600));
 
    std::thread thread_1(blur_img, std::ref(img_lt));
    std::thread thread_2(blur_img, std::ref(img_rt));
    std::thread thread_3(blur_img, std::ref(img_lb));
    std::thread thread_4(blur_img, std::ref(img_rb));
 
    thread_1.join();
    thread_2.join();
    thread_3.join();
    thread_4.join();
 
    cv::hconcat(img_lt, img_rt, img_lt);
    cv::hconcat(img_lb, img_rb, img_lb);
    cv::vconcat(img_lt, img_lb, img);
    return img;
}
 
void mutexTest() {
    std::mutex mtx;
    int c = 0;

    for (int i = 0; i < 100000; i++) {
        std::lock_guard<std::mutex> lock(mtx);
        c++;
    }
}
 
void atomicTest() {
    std::atomic<int> c(0);

    for (int i = 0; i < 100000; i++) {
        c++;
    }
}
 
int main() {
    std::string path = "../../test.jpg";
    cv::Mat img = cv::imread(path);
    cv::resize(img, img, cv::Size(2000, 1200));
 
    // последовательная обработка
    auto start = std::chrono::system_clock::now();
    cv::Mat img_after_blur = sequentialBlur(img);
    auto end = std::chrono::system_clock::now();
    std::cout << "Step-by-step img blur " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ns" << std::endl;
 

    // параллельная обработка
    start = std::chrono::system_clock::now();
    cv::Mat img_after_blur_thread = parallellBlurThreads(img);
    end = std::chrono::system_clock::now();
    std::cout << "Threading img blur: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
 

    // всего изображения сразу
    start = std::chrono::system_clock::now();
    cv::GaussianBlur(img, img, cv::Size(5, 5), 0);
    end = std::chrono::system_clock::now();
    std::cout << "Whole img blur: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;



    // mutex
    start = std::chrono::system_clock::now();
    std::thread threads_mutex[4];

    for (int i = 0; i < 4; ++i) {
        threads_mutex[i] = std::thread(mutexTest);
    }

    for (int i = 0; i < 4; ++i) {
        threads_mutex[i].join();
    }

    end = std::chrono::system_clock::now();
    std::cout << "mutex loop: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
 
    // atomic
    start = std::chrono::system_clock::now();
    std::thread threads_atomic[4];

    for (int i = 0; i < 4; ++i) {
        threads_atomic[i] = std::thread(atomicTest);
    }

    for (int i = 0; i < 4; ++i) {
        threads_atomic[i].join();
    }
    
    end = std::chrono::system_clock::now();
    std::cout << "atomic loop: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    
    cv::imshow("img_after_blur", img_after_blur);
    cv::imshow("img_after_blur_thread", img_after_blur_thread);
    cv::waitKey(0);
}

