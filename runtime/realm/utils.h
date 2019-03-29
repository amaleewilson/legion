/* Copyright 2019 Stanford University, NVIDIA Corporation
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

// little helper utilities for Realm code
// none of this is Realm-specific, but it's put in the Realm namespace to
//  reduce the chance of conflicts

#ifndef REALM_UTILS_H
#define REALM_UTILS_H

#include <string>
#include <ostream>
#include <vector>
#include <map>
#include <cassert>

namespace Realm {
    
  // helpers for deleting contents STL containers of pointers-to-things

  template <typename T>
  void delete_container_contents(std::vector<T *>& v, bool clear_cont = true)
  {
    for(typename std::vector<T *>::iterator it = v.begin();
	it != v.end();
	it++)
      delete (*it);

    if(clear_cont)
      v.clear();
  }

  template <typename K, typename V>
  void delete_container_contents(std::map<K, V *>& m, bool clear_cont = true)
  {
    for(typename std::map<K, V *>::iterator it = m.begin();
	it != m.end();
	it++)
      delete it->second;

    if(clear_cont)
      m.clear();
  }

  // streambuf that holds most messages in an internal buffer
  template <size_t _INTERNAL_BUFFER_SIZE, size_t _INITIAL_EXTERNAL_SIZE>
  class shortstringbuf : public std::streambuf {
  public:
    shortstringbuf();
    ~shortstringbuf();

    const char *data() const;
    size_t size() const;

  protected:
    virtual int_type overflow(int_type c);

    static const size_t INTERNAL_BUFFER_SIZE = _INTERNAL_BUFFER_SIZE;
    static const size_t INITIAL_EXTERNAL_BUFFER_SIZE = _INITIAL_EXTERNAL_SIZE;
    char internal_buffer[INTERNAL_BUFFER_SIZE];
    char *external_buffer;
    size_t external_buffer_size;
  };


  // helper class that lets you build a formatted std::string as a single expression:
  //  /*std::string s =*/ stringbuilder() << ... << ... << ...;

  class stringbuilder {
  public:
    stringbuilder() : os(&strbuf) {}
    operator std::string(void) const { return std::string(strbuf.data(),
							  strbuf.size()); }
    template <typename T>
    stringbuilder& operator<<(T data) { os << data; return *this; }
  protected:
    shortstringbuf<32, 64> strbuf;
    std::ostream os;
  };


  // behaves like static_cast, but uses dynamic_cast+assert when DEBUG_REALM
  //  is defined
  template <typename T, typename T2>
  inline T checked_cast(T2 *ptr)
  {
#ifdef DEBUG_REALM
    T result = dynamic_cast<T>(ptr);
    assert(result != 0);
    return result;
#else
    return static_cast<T>(ptr);
#endif
  }


  // a wrapper class that defers construction of the underlying object until
  //  explicitly requested
  template <typename T>
  class DeferredConstructor {
  public:
    DeferredConstructor();
    ~DeferredConstructor();

    // zero and one argument constructors for now
    T *construct();

    template <typename T1>
    T *construct(T1 arg1);

    // object must have already been explicitly constructed to dereference
    T& operator*();
    T *operator->();

    const T& operator*() const;
    const T *operator->() const;

  protected:
    T *ptr;  // needed to avoid type-punning complaints
    char raw_storage[sizeof(T)] __attribute((aligned(__alignof__(T))));
  };


  template <unsigned _BITS, unsigned _SHIFT>
  struct bitfield {
    static const unsigned BITS = _BITS;
    static const unsigned SHIFT = _SHIFT;

    template <typename T>
    static T extract(T source);

    template <typename T>
    static T insert(T target, T field);

    template <typename T>
    static T bit_or(T target, T field);
  };

  template <typename T>
  class bitpack {
  public:
    bitpack();  // no initialization
    bitpack(const bitpack<T>& copy_from);
    bitpack(T init_val);

    bitpack<T>& operator=(const bitpack<T>& copy_from);
    bitpack<T>& operator=(T new_val);

    operator T() const;

    template <typename BITFIELD>
    class bitsliceref {
    public:
      bitsliceref(T& _target);

      operator T() const;
      bitsliceref<BITFIELD>& operator=(T field);
      bitsliceref<BITFIELD>& operator|=(T field);

    protected:
      T& target;
    };

    template <typename BITFIELD>
    class constbitsliceref {
    public:
      constbitsliceref(const T& _target);

      operator T() const;

    protected:
      const T& target;
    };

    template <typename BITFIELD>
    bitsliceref<BITFIELD> slice();
    template <typename BITFIELD>
    constbitsliceref<BITFIELD> slice() const;

    template <typename BITFIELD>
    bitsliceref<BITFIELD> operator[](const BITFIELD& bitfield);
    template <typename BITFIELD>
    constbitsliceref<BITFIELD> operator[](const BITFIELD& bitfield) const;

  protected:
    T value;
  };

}; // namespace Realm

#include "utils.inl"

#endif // ifndef REALM_UTILS_H
