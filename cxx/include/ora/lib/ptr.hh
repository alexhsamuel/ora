#pragma once

#include <memory>

namespace ora {
namespace lib {

//------------------------------------------------------------------------------

template<typename VAL>
inline VAL*
default_new()
{
  return new VAL;
}


template<typename VAL, VAL* (*ALLOCATE)()=default_new<VAL>>
class LazyPointer
{
public:

  using Value = VAL;
  using Allocator = VAL* (*)();
  static Allocator constexpr allocate = ALLOCATE;

  void 
  initialize()
    const
  {
    if (pointer_ == nullptr) {
      pointer_.reset(allocate());
      assert(pointer_ != nullptr);
    }
  }

  void
  reset(
    Value* val=nullptr)
  {
    pointer_.reset(val);
  }

  Value* 
  get() 
    const 
  { 
    initialize(); 
    return pointer_.get(); 
  }

  Value* operator->() const { return get(); }
  Value& operator*()  const { return *get(); }

private:

  mutable std::unique_ptr<Value> pointer_;

};


//------------------------------------------------------------------------------

}  // namespace lib
}  // namespace ora

