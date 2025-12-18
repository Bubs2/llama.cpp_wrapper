#include "llama_model.h"
#include "llama.h"
#include "mtmd-helper.h"

#include <format>

namespace llama_server::internal {

	// ===================================================================
	// LlamaModel::ModelDeleter
	// ===================================================================

	void LlamaModel::ModelDeleter::operator()(llama_model* model) const { llama_model_free(model); }

	// ===================================================================
	// LlamaModel::MtmdDeleter
	// ===================================================================

	void LlamaModel::MtmdDeleter::operator()(mtmd_context* mtmd) const { mtmd_free(mtmd); }

	// ===================================================================
	// LlamaModel
	// ===================================================================

	LlamaModel::LlamaModel(
		std::string_view model_path,
		const llama_model_params& model_params,
		std::string_view mtmd_path,
		const mtmd_context_params& mtmd_params
	) {
		model_ = std::unique_ptr<llama_model, ModelDeleter>(llama_model_load_from_file(model_path.data(), model_params));
		if (!model_) throw LlamaException(std::format("Failed to load model from file: {}", model_path));

		if (!mtmd_path.empty()) {
			mtmd_ = std::unique_ptr<mtmd_context, MtmdDeleter>(mtmd_init_from_file(mtmd_path.data(), model_.get(), mtmd_params));
			if (!mtmd_) throw LlamaException(std::format("Failed to load mtmd from file: {}", mtmd_path));
		}
	}

	LlamaModel::~LlamaModel() { return; }

	LlamaModel::LlamaModel(LlamaModel&&) noexcept = default;
	LlamaModel& LlamaModel::operator=(LlamaModel&&) noexcept = default;

	const llama_vocab* LlamaModel::get_vocab() const { return llama_model_get_vocab(model_.get()); }

}