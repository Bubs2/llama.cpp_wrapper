#pragma once

#include "mtmd.h"

#include <memory>
#include <vector>
#include <string>

namespace llama_server::internal {

	struct IDChunk {
		struct ChunkDeleter {
			void operator()(mtmd_input_chunk* ptr) { mtmd_input_chunk_free(ptr); }
		};

		std::string id;
		std::unique_ptr<mtmd_input_chunk, ChunkDeleter> chunk;

		IDChunk() = default;
		explicit IDChunk(std::string id, const mtmd_input_chunk* chunk)
			: id(id), chunk(std::unique_ptr<mtmd_input_chunk, ChunkDeleter>(mtmd_input_chunk_copy(chunk)))
		{
		}

		IDChunk(IDChunk&&) = default;
		IDChunk& operator=(IDChunk&&) = default;
		IDChunk(const IDChunk&) = delete;
		IDChunk& operator=(const IDChunk&) = delete;

		const mtmd_input_chunk* get_data() const { return chunk.get(); }
		mtmd_input_chunk_type get_type() const { return mtmd_input_chunk_get_type(chunk.get()); }
		size_t get_n_tokens() const { return mtmd_input_chunk_get_n_tokens(chunk.get()); }
	};

	using IDChunkPtr = std::shared_ptr<IDChunk>;

	struct IDChunkPtrHash {
		using is_transparent = void;
		size_t operator()(const IDChunkPtr& ptr) const {
			if (!ptr) return 0;
			return std::hash<std::string_view>{}(ptr->id);
		}
		size_t operator()(std::string_view id) const {
			return std::hash<std::string_view>{}(id);
		}
	};

	struct IDChunkPtrEqual {
		using is_transparent = void;
		bool operator()(const IDChunkPtr& lhs, const IDChunkPtr& rhs) const {
			if (lhs == rhs) return true;
			if (!lhs || !rhs) return false;
			return lhs->id == rhs->id;
		}
		bool operator()(const IDChunkPtr& lhs, std::string_view rhs) const {
			return lhs && lhs->id == rhs;
		}
		bool operator()(std::string_view lhs, const IDChunkPtr& rhs) const {
			return rhs && lhs == rhs->id;
		}
	};

}