#pragma once

#include "llama_exception.h"

#include <memory>
#include <vector>

struct common_chat_params;

namespace llama_server {

    namespace internal {
        struct ChatParamsAccessor;
        struct ChunksAccessor;

        class LlamaModel;
        class Templater;
        class Tokenizer;
    }

    namespace history_manager_details {
        class MessageManager;
        class ChunkManager;
    }

    struct Message {
        std::string role;
        std::string content;
    };

    struct Tool {
        std::string name;
        std::string description;
        std::string parameters;
    };

    class HistoryManager {
    public:
        HistoryManager(
            std::shared_ptr<internal::LlamaModel> model,
            std::shared_ptr<internal::Tokenizer> tokenizer,
            std::shared_ptr<internal::Templater> templater,
            size_t context_size
        );
        ~HistoryManager();

        void add_message(Message msg, bool is_head_prompt = false);

        void add_tool(Tool tool);
    private:
        std::shared_ptr<internal::Tokenizer> tokenizer_;
        std::shared_ptr<internal::Templater> templater_;

        std::unique_ptr<history_manager_details::MessageManager> message_manager_;
        std::unique_ptr<history_manager_details::ChunkManager> chunk_manager_;

        size_t context_size_;
        std::unique_ptr<::common_chat_params> chat_params_cache_;

        size_t total_token_estimate_ = 0;
        std::vector<size_t> msg_token_counts_;
        size_t tools_token_count_ = 0;

        void unload_cache(size_t spare);
        void estimate_msg_info(size_t index);
        void estimate_tools_info();

        friend struct internal::ChatParamsAccessor;
        friend struct internal::ChunksAccessor;
    };

}