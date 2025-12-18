#pragma once

#include "llama_configs.h"
#include "llama_exception.h"
#include "history_manager.h"

#include <memory>

namespace llama_server {

	namespace internal {
		class LlamaModel;
		class LlamaContext;
		class KVScheduler;
		class Tokenizer;
		class Templater;
		class Sampler;
		class Streamer;
	}
	

	class LlamaSession {
	public:
		LlamaSession(
			ContextConfig context_config,
			std::shared_ptr<internal::LlamaModel> model
		);
		~LlamaSession();

		LlamaSession(const LlamaSession&) = delete;
		LlamaSession& operator=(const LlamaSession&) = delete;
		LlamaSession(LlamaSession&&) noexcept;
		LlamaSession& operator=(LlamaSession&&) noexcept;

		void generate(const GenConfig& gen_params);

		HistoryManager& access_history_manager();
	private:
		std::shared_ptr<internal::LlamaContext> context_;
		std::unique_ptr<internal::KVScheduler> kv_scheduler_;
		std::unique_ptr<internal::Sampler> sampler_;

		std::shared_ptr<internal::Tokenizer> tokenizer_;
		std::shared_ptr<internal::Templater> templater_;

		std::unique_ptr<HistoryManager> history_manager_;

		std::unique_ptr<internal::Streamer> streamer_;
	};

}