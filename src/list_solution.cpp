// Implementation of List<T> methods

template<typename T>
T& List<T>::Back() {
    if (size_ == 0) {
        throw std::out_of_range("Back(): List is empty.");
    }
    return elems_[size_ - 1];
}

template<typename T>
const T& List<T>::Back() const {
    return const_cast<List*>(this)->Back();
}

template<typename T>
void List<T>::PushFront(const T& val) {
    Resize(size_ + 1);
    for (size_t i = size_ - 1; i > 0; i--) {
        elems_[i] = elems_[i - 1];
    }
    elems_[0] = val;
}

template<typename T>
void List<T>::PushFront(T&& val) {
    Resize(size_ + 1);
    for (size_t i = size_ - 1; i > 0; i--) {
        elems_[i] = elems_[i - 1];
    }
    elems_[0] = std::move(val);
}

template<typename T>
void List<T>::PopFront() {
    if (size_ == 0) {
        throw std::out_of_range("PopFront(): List is empty.");
    }
    for (size_t i = 0; i < size_ - 1; i++) {
        elems_[i] = elems_[i + 1];
    }
    Resize(size_ - 1);
}

template<typename T>
bool List<T>::operator==(const List<T>& rhs) const {
    if (size_ != rhs.size_) return false;
    for (size_t i = 0; i < size_; i++) {
        if (elems_[i] != rhs.elems_[i]) return false;
    }
    return true;
}

template<typename T>
bool List<T>::operator!=(const List<T>& rhs) const {
    return !(*this == rhs);
}

template<typename T>
void List<T>::Remove(const T& val) {
    size_t newSize = 0;
    for (size_t i = 0; i < size_; i++) {
        if (elems_[i] != val) {
            elems_[newSize] = elems_[i];
            newSize++;
        }
    }
    Resize(newSize);
}

template<typename T>
void List<T>::Fill(const T& v) {
    for (size_t i = 0; i < size_; i++) {
        elems_[i] = v;
    }
}

template<typename T>
void List<T>::Clear() noexcept {
    size_ = 0;
}

template<typename T>
void List<T>::Insert(size_t pos, const T& val) {
    if (pos > size_) {
        throw std::out_of_range("Insert(pos, val): " + std::to_string(pos) + " > " + std::to_string(size_));
    }
    Resize(size_ + 1);
    for (size_t i = size_ - 1; i > pos; i--) {
        elems_[i] = elems_[i - 1];
    }
    elems_[pos] = val;
}

template<typename T>
void List<T>::Insert(size_t pos, T&& val) {
    if (pos > size_) {
        throw std::out_of_range("Insert(pos, val): " + std::to_string(pos) + " > " + std::to_string(size_));
    }
    Resize(size_ + 1);
    for (size_t i = size_ - 1; i > pos; i--) {
        elems_[i] = elems_[i - 1];
    }
    elems_[pos] = std::move(val);
}

template<typename T>
void List<T>::Erase(size_t pos) {
    if (pos >= size_) {
        throw std::out_of_range("Erase(pos): " + std::to_string(pos) + " >= " + std::to_string(size_));
    }
    for (size_t i = pos; i < size_ - 1; i++) {
        elems_[i] = elems_[i + 1];
    }
    Resize(size_ - 1);
}

template<typename T>
void List<T>::Swap(List<T>& other) {
    std::swap(elems_, other.elems_);
    std::swap(size_, other.size_);
    std::swap(capacity_, other.capacity_);
}

template<typename T>
void List<T>::Assign(size_t count, const T& val) {
    Resize(count);
    for (size_t i = 0; i < count; i++) {
        elems_[i] = val;
    }
}
