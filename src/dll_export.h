#pragma once

#if defined _WIN32 || defined __CYGWIN__
#ifdef BUILDLIB
#define DLLEXPORT __declspec(dllexport)
#define WEAKDLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __declspec(dllimport)
#define WEAKDLLEXPORT
#endif
#else
#define DLLEXPORT
#define WEAKDLLEXPORT
#endif