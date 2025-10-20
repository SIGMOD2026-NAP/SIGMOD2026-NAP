#pragma once

#define __SSE__
#define __AVX__


#ifndef NO_MANUAL_VECTORIZATION
#ifdef __SSE__
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#endif
#endif
#endif

#if defined(USE_AVX) || defined(USE_SSE)
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>
#else
#include <x86intrin.h>
#endif

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif
#endif

#include <queue>
#include <vector>
#include <iostream>
#include <string.h>
#include <atomic>

extern std::atomic<size_t> _G_COST;

namespace hnswlib {
    typedef size_t labeltype;

    template <typename T>
    class pairGreater {
        public:
        bool operator()(const T& p1, const T& p2) {
            return p1.first > p2.first;
        }
    };

    template<typename T>
    static void writeBinaryPOD(std::ostream& out, const T& podRef) {
        out.write((char*)&podRef, sizeof(T));
    }

    template<typename T>
    static void readBinaryPOD(std::istream& in, T& podRef) {
        in.read((char*)&podRef, sizeof(T));
    }

    template<typename MTYPE>
    using DISTFUNC = MTYPE(*)(const void*, const void*, const void*);

    //template<typename MTYPE>
    //using DISTFUNC = MTYPE(*)(float*, float*, int);

    template<typename MTYPE>
    class SpaceInterface {
        public:
        //virtual void search(void *);
        virtual size_t get_data_size() = 0;

        virtual DISTFUNC<MTYPE> get_dist_func() = 0;

        virtual void* get_dist_func_param() = 0;

        virtual ~SpaceInterface() {}
    };

    template<typename dist_t>
    class AlgorithmInterface {
        public:
        //virtual void addPoint(const void *datapoint, labeltype label)=0;
        //virtual std::priority_queue<std::pair<dist_t, labeltype >> searchKnn(const void *, size_t) const = 0;
        template <typename Comp>
        std::vector<std::pair<dist_t, labeltype>> searchKnn(const void*, size_t, Comp) {
        }
        virtual void saveIndex(const std::string& location) = 0;
        virtual ~AlgorithmInterface() {
        }
    };


}

template<typename MTYPE>
using DISTFUNC = MTYPE(*)(const void*, const void*, const void*);
//using namespace hnswlib;

#include "basis.h"

static float cal_inner_product_hnsw(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    //++_G_COST;
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);

    return 1.0f - cal_inner_product(pVect1, pVect2, qty);
}

static float cal_L2sqr_hnsw(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    //++cost;
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);

    return cal_L2sqr(pVect1, pVect2, qty);
}

class IpSpace : public hnswlib::SpaceInterface<float> {

    DISTFUNC<float> fstdistfunc_ = cal_inner_product_hnsw;
    size_t data_size_;
    size_t dim_;
    public:
    IpSpace(size_t dim) {
        fstdistfunc_ = cal_inner_product_hnsw;
        //#if defined(USE_SSE) || defined(USE_AVX)
        //        if (dim % 16 == 0)
        //            fstdistfunc_ = L2SqrSIMD16Ext;
        //        else if (dim % 4 == 0)
        //            fstdistfunc_ = L2SqrSIMD4Ext;
        //        else if (dim > 16)
        //            fstdistfunc_ = L2SqrSIMD16ExtResiduals;
        //        else if (dim > 4)
        //            fstdistfunc_ = L2SqrSIMD4ExtResiduals;
        //#endif
        dim_ = dim;
        data_size_ = dim * sizeof(float);
    }

    size_t get_data_size() {
        return data_size_;
    }

    hnswlib::DISTFUNC<float> get_dist_func() {
        return reinterpret_cast<hnswlib::DISTFUNC<float>>(fstdistfunc_);
    }

    void* get_dist_func_param() {
        return &dim_;
    }

    ~IpSpace() {}
};

class L2Space : public hnswlib::SpaceInterface<float> {

    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;
    public:
    L2Space(size_t dim) {
        fstdistfunc_ = cal_L2sqr_hnsw;
        //#if defined(USE_SSE) || defined(USE_AVX)
        //        if (dim % 16 == 0)
        //            fstdistfunc_ = L2SqrSIMD16Ext;
        //        else if (dim % 4 == 0)
        //            fstdistfunc_ = L2SqrSIMD4Ext;
        //        else if (dim > 16)
        //            fstdistfunc_ = L2SqrSIMD16ExtResiduals;
        //        else if (dim > 4)
        //            fstdistfunc_ = L2SqrSIMD4ExtResiduals;
        //#endif
        dim_ = dim;
        data_size_ = dim * sizeof(float);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() {
        return fstdistfunc_;
    }

    void* get_dist_func_param() {
        return &dim_;
    }

    ~L2Space() {}
};

//#include "space_l2.h"
//#include "space_ip.h"
// #include "bruteforce.h"
#include "hnswalg.h"
