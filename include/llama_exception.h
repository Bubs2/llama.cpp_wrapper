#pragma once

#include <stdexcept>

namespace llama_server {

	class LlamaException : public std::runtime_error {
	public:
		using std::runtime_error::runtime_error;
	};

}