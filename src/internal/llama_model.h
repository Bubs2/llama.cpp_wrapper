#pragma once

#include "llama_exception.h"

#include <memory>
#include <string_view>

using llama_token = int32_t;

struct llama_model_params;
struct llama_model;
struct llama_vocab;

struct mtmd_context_params;
struct mtmd_context;

namespace llama_server::internal {

	class LlamaModel {
	public:
		LlamaModel(
			std::string_view model_path,
			const llama_model_params& model_params,
			std::string_view mtmd_path,
			const mtmd_context_params& mtmd_params
		);
		~LlamaModel();

		LlamaModel(const LlamaModel&) = delete;
		LlamaModel& operator=(const LlamaModel&) = delete;
		LlamaModel(LlamaModel&&) noexcept;
		LlamaModel& operator=(LlamaModel&&) noexcept;

		llama_model* get_data() const { return model_.get(); }
		mtmd_context* get_mtmd() const { return mtmd_.get(); }
		const llama_vocab* get_vocab() const;
	private:
		struct ModelDeleter {
			void operator()(llama_model* model) const;
		};

		struct MtmdDeleter {
			void operator()(mtmd_context* mtmd) const;
		};

		std::unique_ptr<llama_model, ModelDeleter> model_;
		std::unique_ptr<mtmd_context, MtmdDeleter> mtmd_ = nullptr;
	};

}