#include <string>


extern "C" {
    void* std_string_init(char* in_str, int64_t size) {
        return new std::string(in_str, size);
    }

    const char* std_string_get_cstr(std::string* s) {
        return s->c_str();
    }

    void* std_string_concat(std::string* left, std::string* right) {
        std::string* res = new std::string((*left) + (*right));
        return res;
    }
}