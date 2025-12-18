#include "model_server.h"
#include "llama_model.h"
#include "llama_log.h"
#include "llama.h"
#include "mtmd.h"

#include <format>

namespace llama_server {

    using namespace internal;

    // ===================================================================
    // ModelServer
    // ===================================================================

    ModelServer::ModelServer() {}
    ModelServer::~ModelServer() {}

    ModelServer& ModelServer::get_server() {
        static ModelServer server = ModelServer();
        return server;
    }

    void ModelServer::shutdown() {
        if (shutdown_flag_) return;

        shutdown_flag_ = true;

        std::unique_lock lock(mutex_);

        auto delete_queue = std::move(model_map_);
        model_map_.clear();
        loading_model_set_.clear();
        loading_model_cv_.notify_all();

        lock.unlock();

        delete_queue.clear();
    }

    void ModelServer::load_model(
        const ModelConfig& config,
        std::string name
    ) {
        if (shutdown_flag_) {
            throw ServerShutdownException("ModelServer is shutdown. Cannot load model: " + name);
        }

        std::unique_lock lock(mutex_);

        if (shutdown_flag_) {
            throw ServerShutdownException("ModelServer is shutdown. Cannot load model: " + name);
        }

        if (model_map_.find(name) != model_map_.end()) return;

        while (loading_model_set_.find(name) != loading_model_set_.end()) {
            log_info(std::format("ModelServer: Waiting for model loading: {}", name));
            loading_model_cv_.wait(lock);

            if (shutdown_flag_) throw ServerShutdownException("ModelServer shutdown while loading model: " + name);
            if (model_map_.find(name) != model_map_.end()) return;
        }

        loading_model_set_.emplace(name);

        lock.unlock();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = config.n_gpu_layers;
        model_params.use_mmap = config.use_mmap;
        model_params.use_mlock = config.use_mlock;

        mtmd_context_params mtmd_params = mtmd_context_params_default();
        mtmd_params.image_min_tokens = config.image_min_tokens;
        mtmd_params.image_max_tokens = config.image_max_tokens;

        std::shared_ptr<LlamaModel> model;

        try { model = std::make_shared<LlamaModel>(config.model_path, model_params, config.mtmd_path, mtmd_params); }
        catch (const LlamaException& e) {
            log_error(e.what());
            lock.lock();
            loading_model_set_.erase(name);
            loading_model_cv_.notify_all();
            throw LlamaException("ModelServer: Failed to load model: " + name);
        }

        lock.lock();

        if (shutdown_flag_) {
            loading_model_set_.erase(name);
            loading_model_cv_.notify_all();
            throw ServerShutdownException("ModelServer is shutdown after loading model: " + name);
        }

        loading_model_set_.erase(name);
        model_map_.emplace(std::move(name), std::move(model));
        loading_model_cv_.notify_all();
    }

    void ModelServer::unload_model(
        std::string name
    ) {
        if (shutdown_flag_) {
            throw ServerShutdownException("ModelServer is shutdown. Cannot load model: " + name);
        }

        std::unique_lock lock(mutex_);

        if (shutdown_flag_) {
            throw ServerShutdownException("ModelServer is shutdown. Cannot get model: " + name);
        }

        if (loading_model_set_.find(name) != loading_model_set_.end()) {
            throw UnloadWhenLoadingModelException("Model is loading, try unload after loading is done: " + name);
        }

        model_map_.erase(name);
    }

    std::unique_ptr<LlamaSession> ModelServer::get_session(
        std::string model_name,
        ContextConfig context_config
    ) const {
        std::shared_lock lock(mutex_);

        if (shutdown_flag_) {
            throw ServerShutdownException("ModelServer is shutdown. Cannot get session.");
        }

        auto model = model_map_.find(model_name);

        while (loading_model_set_.find(model_name) != loading_model_set_.end()) {
            log_info(std::format("ModelServer: Waiting for model loading: {}", model_name));
            loading_model_cv_.wait(lock);

            if (shutdown_flag_) throw ServerShutdownException("ModelServer shutdown while loading model: " + model_name);
            if ((model = model_map_.find(model_name)) != model_map_.end()) break;
        }

        if (model == model_map_.end()) {
            throw LlamaException("Model not found: " + model_name);
        }

        return std::make_unique<LlamaSession>(context_config, model->second);
    }

}