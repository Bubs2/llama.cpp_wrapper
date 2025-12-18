#pragma once

#include <spdlog/spdlog.h>
#include <source_location>
#include <string_view>
#include <format>

namespace llama_server::internal {

    namespace llama_log_details {
        inline void log_impl(spdlog::level::level_enum lvl, std::string_view msg, const std::source_location& loc) {
            spdlog::log(lvl, "[{}:{}] [{}] {}",
                loc.file_name(),
                loc.line(),
                loc.function_name(),
                msg);
        }
    }

    inline void log_info(std::string_view msg,
        const std::source_location& loc = std::source_location::current()) {
        llama_log_details::log_impl(spdlog::level::info, msg, loc);
    }

    inline void log_warn(std::string_view msg,
        const std::source_location& loc = std::source_location::current()) {
        llama_log_details::log_impl(spdlog::level::warn, msg, loc);
    }

    inline void log_error(std::string_view msg,
        const std::source_location& loc = std::source_location::current()) {
        llama_log_details::log_impl(spdlog::level::err, msg, loc);
    }

}