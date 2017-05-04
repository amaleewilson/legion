/* Copyright 2017 Stanford University, NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#ifndef COMMON_H
#define COMMON_H

// This file contains declarations of objects
// that need to be globally visible to all layers
// of the program including the application code
// as well as both of the runtimes.

struct ptr_t
{ 
public:
#ifdef __CUDACC__
  __host__ __device__
#endif
  ptr_t(void) : value(0) { }
#ifdef __CUDACC__
  __host__ __device__
#endif
  ptr_t(const ptr_t &p) : value(p.value) { }
#ifdef __CUDACC__
  __host__ __device__
#endif
  ptr_t(long long int v) : value(v) { }
public:
  long long int value; 
public: 
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline ptr_t& operator=(const ptr_t &ptr) { value = ptr.value; return *this; }
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline bool operator==(const ptr_t &ptr) const { return (ptr.value == this->value); }
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline bool operator!=(const ptr_t &ptr) const { return (ptr.value != this->value); }
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline bool operator< (const ptr_t &ptr) const { return (ptr.value <  this->value); }
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline operator bool(void) const { return (value != -1LL); }
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline bool operator!(void) const { return (value == -1LL); }

  // Addition operation on pointers
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline ptr_t operator+(const ptr_t &ptr) const { return ptr_t(value + ptr.value); }
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline ptr_t operator+(unsigned offset) const { return ptr_t(value + offset); }
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline ptr_t operator+(int offset) const { return ptr_t(value + offset); }

  // Subtraction operation on pointers
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline ptr_t operator-(const ptr_t &ptr) const { return ptr_t(value - ptr.value); }
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline ptr_t operator-(unsigned offset) const { return ptr_t(value - offset); }
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline ptr_t operator-(int offset) const { return ptr_t(value - offset); }
  
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline ptr_t& operator++(void) { value++; return *this; }
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline ptr_t operator++(int) { value++; return *this; }
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline ptr_t& operator--(void) { value--; return *this; }
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline ptr_t operator--(int) { value--; return *this; }

  // Thank you Eric for type cast operators!
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline operator long long int(void) const { return value; }

#ifdef __CUDACC__
  __host__ __device__
#endif
  inline bool is_null(void) const { return (value == -1LL); }

#ifdef __CUDACC__
  __host__ __device__ 
#endif
  static inline ptr_t nil(void) { ptr_t p; p.value = -1LL; return p; }
};

#endif // COMMON_H
