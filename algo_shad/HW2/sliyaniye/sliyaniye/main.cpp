//
//  main.cpp
//  sliyaniye
//
//  Created by Simon Fedotov on 19.10.2017.
//  Copyright © 2017 Simon23Rus. All rights reserved.
//

#include <iostream>
#include <algorithm>
#include <vector>

struct ElemWithInd {
    int value;
    int index;
    
    ElemWithInd(int val, int ind) {
        value = val;
        index = ind;
    }
};

class MyHeap {
public:
    std::vector<ElemWithInd> curData;

    explicit MyHeap(std::vector<ElemWithInd> &givenData) {
        curData = givenData;
        
        for (int i = static_cast<int>(givenData.size() / 2); i >= 0; --i) {
            siftDown(i);
        }
    }
    
    void siftDown(int index) {
        int left = 2 * index + 1;
        int right = 2 * index + 2;
        
        if (left >= static_cast<int>(curData.size()))
            return;
        int minChild = left;
        if (right < static_cast<int>(curData.size())) {
            if (curData[right].value < curData[left].value) {
                minChild = right;
            }
        }
        if (curData[index].value > curData[minChild].value) {
            std::swap(curData[index], curData[minChild]);
            siftDown(minChild);
        }
    }
   
    ElemWithInd extractMin() {
        ElemWithInd minimal = curData.front();
        std::swap(curData.front(), curData.back());
        curData.pop_back();
        siftDown(0);
        return minimal;
    }

    void siftUp(int index) {
        if (index == 0) {
            return;
        }
        
        int parent = (index - 1) / 2;
        if (curData[index].value < curData[parent].value) {
            std::swap(curData[index], curData[parent]);
            siftUp(parent);
        }
    }
    
    void insert(int value, int index) {
        curData.push_back(ElemWithInd(value, index));
        siftUp(static_cast<int>(curData.size() - 1));
    }
};

int main(int argc, const char * argv[]) {
    std::ios_base::sync_with_stdio(false);
    int arraysNum, lenOfArray;
    std::cin >> arraysNum >> lenOfArray;
    std::vector<std::vector<int>> data(arraysNum, std::vector<int>(lenOfArray));
    std::vector<int> answer;
    for (int i = 0; i < arraysNum; ++i) {
        for (int j = 0; j < lenOfArray; ++j) {
            int number;
            std::cin >> number;
            data[i][j] = number;
        }
    }
    
    
    std::vector<ElemWithInd> firstPortion;
    for (int i = 0; i < static_cast<int>(data.size()); ++i) {
        firstPortion.push_back(ElemWithInd(data[i][0], i));
    }
    
    std::vector<int> posInArray(arraysNum, 1);
    MyHeap kutcha = MyHeap(firstPortion);
    int insertionsNumber = 0;
    while (insertionsNumber < arraysNum * (lenOfArray - 1)) {
        ElemWithInd curMin = kutcha.extractMin();
        int whereToExtract = curMin.index;
        if (posInArray[whereToExtract] < lenOfArray) {
            kutcha.insert(data[whereToExtract][posInArray[whereToExtract]], whereToExtract);
            ++posInArray[whereToExtract];
            ++insertionsNumber;
        }
        answer.push_back(curMin.value);
    }
    
    while (!kutcha.curData.empty()) {
        ElemWithInd curMin = kutcha.extractMin();
        answer.push_back(curMin.value);
    }
    for (auto elem : answer) {
        std::cout << elem << " ";
    }
    return 0;
}
