#pragma once

#include <functional>

namespace llama_server::internal {

	template <typename TOwner, typename TBorrower = TOwner>
	class LoanGuard {
	public:
		LoanGuard(TOwner& owner_ref, TBorrower& borrower_ref)
			: owner_(owner_ref), borrower_(borrower_ref) {
			if constexpr (std::is_assignable_v<TBorrower&, TOwner>) borrower_ = std::move(owner_);
			else static_assert(false,
				"LoanGuard: Default constructor used but types are not assignable. You must use the constructor with callbacks.");
		}

		LoanGuard(
			TOwner& owner_ref, TBorrower& borrower_ref,
			std::function<void(TOwner& owner, TBorrower& borrower)> loan_func,
			std::function<void(TOwner& owner, TBorrower& borrower)> return_func
		) : owner_(owner_ref), borrower_(borrower_ref), loan_func_(loan_func), return_func_(return_func) {
			loan_func_(owner_, borrower_);
		}

		~LoanGuard() noexcept {
			if (return_func_) return_func_(owner_, borrower_);
			else if constexpr (std::is_assignable_v<TOwner&, TBorrower>) owner_ = std::move(borrower_);
		}

		LoanGuard(const LoanGuard&) = delete;
		LoanGuard& operator=(const LoanGuard&) = delete;
		LoanGuard(LoanGuard&&) = delete;
		LoanGuard& operator=(LoanGuard&&) = delete;
	private:
		TOwner& owner_;
		TBorrower& borrower_;

		std::function<void(TOwner& owner, TBorrower& borrower)> loan_func_ = nullptr;
		std::function<void(TOwner& owner, TBorrower& borrower)> return_func_ = nullptr;
	};

}