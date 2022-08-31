#include <Python.h>

#include "_std_string.hpp"

#define REGISTER(func) PyObject_SetAttrString(m, #func, PyLong_FromVoidPtr((void*)(&func)));

extern "C" {
    PyMODINIT_FUNC PyInit_std_string(void) {
        PyObject* m;
        static struct PyModuleDef moduleDef = {
                PyModuleDef_HEAD_INIT,
                "_std_string",
                "No docs",
                -1,
                NULL,
        };
        m = PyModule_Create(&moduleDef);
        if (m == NULL) {
            return 0;
        }

        // register init, getter, setter functions
        REGISTER(std_string_init)
        REGISTER(std_string_get_cstr)

        // register string operations
        REGISTER(std_string_concat)

        return m;
    }
}

#undef REGISTER