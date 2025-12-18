#pragma once

#include <string>
#include <string_view>
#include <functional>

namespace llama_server::internal {

	class Streamer {
	public:
		Streamer();
		~Streamer() = default;

		using StreamCallback = std::function<bool(std::string&&)>;
		bool process(std::string_view bytes, StreamCallback callback);
		void clear();
	private:
		std::string buffer_;

		bool validate_utf8_end(std::string_view buffer);
	};

}
