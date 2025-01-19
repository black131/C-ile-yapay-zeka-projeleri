
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
#include <tuple>
#include <iomanip>

// KNN için gerekli kütüphaneler
#include <map>
#include <limits>

// Veri seti oluşturma
std::vector<std::pair<double, double>> generateData() {
    std::vector<std::pair<double, double>> data;
    std::mt19937 generator(0);
    std::uniform_real_distribution<double> dist(0.0, 5.0);
    for (int i = 0; i < 40; ++i) {
        double x = dist(generator);
        double y = std::sin(x);
        if (i % 5 == 0) {
            y += (0.5 - dist(generator) / 5.0);
        }
        data.emplace_back(x, y);
    }
    std::sort(data.begin(), data.end());
    return data;
}

// KNN regressor
double knnRegression(const std::vector<std::pair<double, double>>& data, double target, int k, bool weighted) {
    std::vector<std::pair<double, double>> distances;
    for (const auto& point : data) {
        double dist = std::abs(point.first - target);
        distances.emplace_back(dist, point.second);
    }
    std::sort(distances.begin(), distances.end());
    
    double prediction = 0.0;
    if (weighted) {
        double weight_sum = 0.0;
        for (int i = 0; i < k; ++i) {
            double weight = 1.0 / (distances[i].first + 1e-5);
            prediction += weight * distances[i].second;
            weight_sum += weight;
        }
        prediction /= weight_sum;
    } else {
        for (int i = 0; i < k; ++i) {
            prediction += distances[i].second;
        }
        prediction /= k;
    }
    return prediction;
}

// Mean squared error hesaplama
double meanSquaredError(const std::vector<double>& actual, const std::vector<double>& predicted) {
    double mse = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        mse += std::pow(predicted[i] - actual[i], 2);
    }
    return mse / actual.size();
}

// Mean absolute error hesaplama
double meanAbsoluteError(const std::vector<double>& actual, const std::vector<double>& predicted) {
    double mae = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        mae += std::abs(predicted[i] - actual[i]);
    }
    return mae / actual.size();
}

int main() {
    auto data = generateData();
    
    std::vector<double> X, y;
    for (const auto& point : data) {
        X.push_back(point.first);
        y.push_back(point.second);
    }
    
    std::vector<double> predictions_uniform, predictions_weighted;
    int k = 5;
    for (double target = 0.0; target <= 5.0; target += 0.01) {
        predictions_uniform.push_back(knnRegression(data, target, k, false));
        predictions_weighted.push_back(knnRegression(data, target, k, true));
    }

    std::cout << "MSE and MAE for uniform weighting:" << std::endl;
    double mse_uniform = meanSquaredError(y, predictions_uniform);
    double mae_uniform = meanAbsoluteError(y, predictions_uniform);
    std::cout << "MSE: " << mse_uniform << ", MAE: " << mae_uniform << std::endl;

    std::cout << "MSE and MAE for distance weighting:" << std::endl;
    double mse_weighted = meanSquaredError(y, predictions_weighted);
    double mae_weighted = meanAbsoluteError(y, predictions_weighted);
    std::cout << "MSE: " << mse_weighted << ", MAE: " << mae_weighted << std::endl;

    return 0;
}
