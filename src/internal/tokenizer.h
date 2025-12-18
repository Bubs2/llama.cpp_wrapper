#pragma once

#include "llama_exception.h"
#include "llama.h"
#include "mtmd.h"

#include <memory>
#include <vector>
#include <span>
#include <string>
#include <string_view>

namespace llama_server::internal {

	class LlamaModel;

	class Tokenizer {
	public:
		Tokenizer(std::shared_ptr<LlamaModel> model);
		~Tokenizer();

		std::vector<llama_token> tokenize(
			std::string_view text,
			bool add_special = true,
			bool parse_special = true
		) const;

		std::string detokenize(
			llama_token token,
			bool add_special = false
		) const;

		mtmd::input_chunks_ptr mtmd_tokenize(
			std::string_view text,
			mtmd::bitmaps& bitmaps,
			bool add_special = true,
			bool parse_special = true
		) const;
	private:
		std::shared_ptr<LlamaModel> model_;
	};

}