#pragma once

#include "chat.h"

#include <memory>

namespace llama_server::internal {

	class LlamaModel;

	class Templater {
	public:
		Templater(std::shared_ptr<LlamaModel> model);
		~Templater();

		common_chat_params apply_templates(const common_chat_templates_inputs& inputs) const;
	private:
		common_chat_templates_ptr tmpl_;
	};

}