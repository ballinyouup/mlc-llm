/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file support/debug_utils.h
 * \brief Tools for debug purposes.
 */
#ifndef MLC_LLM_SUPPORT_DEBUG_UTILS_H_
#define MLC_LLM_SUPPORT_DEBUG_UTILS_H_

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/string.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <string>
#include "../tokenizers/tokenizers.h"

namespace mlc {
namespace llm {

using namespace tvm::runtime;

/*! \brief A registry for debug information. */
class DebugRegistry {
 public:
  static DebugRegistry* Global() {
    static DebugRegistry reg;
    return &reg;
  }

  // Tokenizer information, helpful for converting token id to token string in debugging
  Tokenizer tokenizer;
};

/*! \brief Register the tokenizer to the global tokenizer registry. */
inline void DebugRegisterTokenizer(const Tokenizer& tokenizer) {
  DebugRegistry::Global()->tokenizer = tokenizer;
}

/*! \brief Get the registered tokenizer from the global tokenizer registry. */
inline Tokenizer DebugGetTokenizer() { return DebugRegistry::Global()->tokenizer; }

/*! \brief Check if logit debugging is enabled via environment variable. */
inline bool IsLogitDebugEnabled() {
  return std::getenv("MLC_DEBUG_LOGITS") != nullptr;
}

/*! \brief Check if verbose logit debugging is enabled. */
inline bool IsLogitDebugVerbose() {
  const char* debug_level = std::getenv("MLC_DEBUG_LOGITS");
  return debug_level != nullptr && std::string(debug_level) == "verbose";
}

/*! 
 * \brief Print top-k logits for debugging purposes.
 * \param logits The logits NDArray with shape (batch_size, vocab_size)
 * \param phase The phase name (e.g., "PREFILL", "DECODE")
 * \param request_ids The request IDs for each sequence
 * \param top_k Number of top logits to print (default: 10)
 */
inline void DebugPrintLogits(const NDArray& logits, const std::string& phase, 
                            const Array<String>& request_ids, int top_k = 10) {
  if (!IsLogitDebugEnabled()) return;
  
  // Copy logits to CPU for inspection
  NDArray logits_cpu = logits.CopyTo(DLDevice{kDLCPU, 0});
  
  int batch_size = logits_cpu->shape[0];
  int vocab_size = logits_cpu->shape[1];
  const float* logits_data = static_cast<const float*>(logits_cpu->data);
  
  bool verbose = IsLogitDebugVerbose();
  Tokenizer tokenizer = DebugGetTokenizer();
  bool has_tokenizer = tokenizer.defined();
  
  std::cout << "\n=== MLC DEBUG LOGITS [" << phase << "] ===" << std::endl;
  
  for (int seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
    std::string req_id = (seq_idx < request_ids.size()) ? 
                        std::string(request_ids[seq_idx]) : "unknown";
    
    std::cout << "Request ID: " << req_id << " (seq " << seq_idx << ")" << std::endl;
    
    // Get logits for this sequence
    const float* seq_logits = logits_data + seq_idx * vocab_size;
    
    // Create vector of (token_id, logit_value) pairs
    std::vector<std::pair<int, float>> token_logits;
    for (int i = 0; i < vocab_size; ++i) {
      token_logits.emplace_back(i, seq_logits[i]);
    }
    
    // Sort by logit value (descending)
    std::partial_sort(token_logits.begin(), token_logits.begin() + std::min(top_k, vocab_size),
                     token_logits.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Print top-k logits
    int print_count = std::min(top_k, vocab_size);
    for (int i = 0; i < print_count; ++i) {
      int token_id = token_logits[i].first;
      float logit_val = token_logits[i].second;
      
      std::cout << "  [" << i << "] token_id=" << token_id 
                << " logit=" << std::fixed << std::setprecision(4) << logit_val;
      
      // Add token string if tokenizer is available
      if (has_tokenizer && verbose) {
        try {
          std::string token_str = tokenizer->IdToToken(token_id);
          // Escape special characters for readability
          std::string escaped_token;
          for (char c : token_str) {
            if (c == '\n') escaped_token += "\\n";
            else if (c == '\t') escaped_token += "\\t";
            else if (c == '\r') escaped_token += "\\r";
            else if (c < 32 || c > 126) escaped_token += "\\x" + std::to_string((unsigned char)c);
            else escaped_token += c;
          }
          std::cout << " token=\"" << escaped_token << "\"";
        } catch (...) {
          // Ignore tokenizer errors
        }
      }
      std::cout << std::endl;
    }
    
    if (verbose) {
      // Print some statistics
      float max_logit = *std::max_element(seq_logits, seq_logits + vocab_size);
      float min_logit = *std::min_element(seq_logits, seq_logits + vocab_size);
      float sum_logits = std::accumulate(seq_logits, seq_logits + vocab_size, 0.0f);
      float mean_logit = sum_logits / vocab_size;
      
      std::cout << "  Stats: max=" << std::fixed << std::setprecision(4) << max_logit
                << " min=" << min_logit << " mean=" << mean_logit << std::endl;
    }
    
    if (seq_idx < batch_size - 1) {
      std::cout << std::endl;
    }
  }
  
  std::cout << "=== END DEBUG LOGITS ===" << std::endl << std::endl;
}

}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SUPPORT_DEBUG_UTILS_H_
