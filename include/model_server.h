#pragma once

#include "llama_exception.h"
#include "llama_configs.h"
#include "llama_session.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <atomic>
#include <shared_mutex>
#include <condition_variable>

namespace llama_server {

	namespace internal {
		class LlamaModel;
	}

	class ModelServer {
	public:
		static ModelServer& get_server();
		void shutdown();

		ModelServer(const ModelServer&) = delete;
		ModelServer& operator=(const ModelServer&) = delete;
		ModelServer(ModelServer&&) = delete;
		ModelServer& operator=(ModelServer&&) = delete;

		void load_model(const ModelConfig& config, std::string name);
		void unload_model(std::string name);

		std::unique_ptr<LlamaSession> get_session(
			std::string model_name,
			ContextConfig context_config
		) const;

	private:
		ModelServer();
		~ModelServer();

		mutable std::shared_mutex mutex_;
		mutable std::condition_variable_any loading_model_cv_;
		std::atomic<bool> shutdown_flag_;

		std::unordered_map<std::string, std::shared_ptr<internal::LlamaModel>> model_map_;
		std::unordered_set<std::string> loading_model_set_;
	};

	class ServerShutdownException : public LlamaException {
	public:
		using LlamaException::LlamaException;
	};

	class UnloadWhenLoadingModelException : public LlamaException {
	public:
		using LlamaException::LlamaException;
	};

}