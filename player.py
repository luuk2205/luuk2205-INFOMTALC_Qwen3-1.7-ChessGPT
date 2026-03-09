import random
import re

import chess
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from chess_tournament import Player

MOVE_RE = re.compile(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b")

piece_values = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

piece_names = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


def find_uci(text):
    # just pull the first uci-looking move out of the model output
    if not text:
        return None
    match = MOVE_RE.search(text.lower())
    return match.group(0) if match else None


def short_fen(board):
    # i only care about board state here, not move counters
    return " ".join(board.fen().split()[:4])


class TransformerPlayer(Player):
    def __init__(
        self,
        name="Luuk_Kwee_7532784",
        model_id="Qwen/Qwen3-1.7B-Base",
        adapter_repo="luuk2205/Qwen3-1.7B-ChessGPT",
        tries=6,
        verbose_tactics=False,
        use_scholars_mate=True,
    ):
        super().__init__(name)

        self.tries = tries
        self.verbose_tactics = verbose_tactics
        self.use_scholars_mate = use_scholars_mate

        # use 4-bit quantization on CUDA; fall back to regular loading elsewhere.
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if torch.cuda.is_available():
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=(
                    torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16
                ),
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = quant_cfg
        elif torch.backends.mps.is_available():
            model_kwargs["torch_dtype"] = torch.float16

        # base model first, then attach the fine-tuned adapter
        base = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )
        self.model = PeftModel.from_pretrained(base, adapter_repo)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_repo,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # keeping a bit of game memory helps with repetition and logging
        self.pos_counts = {}
        self.my_color = None
        self.last_after_move_board = None

    def reset_state(self):
        # reset everything when a fresh game starts
        self.pos_counts = {}
        self.my_color = None
        self.last_after_move_board = None

    def maybe_reset_game(self, board):
        # rough check for a brand new game
        if board.fullmove_number == 1 and len(board.piece_map()) == 32:
            self.reset_state()

    def build_prompt(self, fen):
        # keeping the prompt strict so the model just outputs one move
        return (
            "You are a high-ELO chess engine playing in the style of Garry Kasparov: "
            "combine deep calculation with relentless pressure, punish any inaccuracy, "
            "prefer dynamic piece activity over material, play for the initiative and "
            "never allow the opponent to consolidate.\n"
            "Given the FEN position, output exactly one UCI move (e.g. e2e4, g1f3, e7e8q).\n"
            "Output only the move, no explanation, no punctuation, no newlines.\n"
            f"FEN: {fen}\n"
            "UCI move:"
        )

    def random_legal_move(self, fen):
        # last-resort fallback so the bot never crashes a turn
        board = chess.Board(fen)
        legal = list(board.legal_moves)
        return random.choice(legal).uci() if legal else None

    def material_balance(self, board, color):
        # simple material count from one side's perspective
        total = 0
        for piece_type, value in piece_values.items():
            ours = len(board.pieces(piece_type, color))
            theirs = len(board.pieces(piece_type, not color))
            total += value * (ours - theirs)
        return total

    def captured_piece(self, board, move):
        # needed because en passant capture square is weird
        if not board.is_capture(move):
            return None, None

        if board.is_en_passant(move):
            if board.turn == chess.WHITE:
                square = move.to_square - 8
            else:
                square = move.to_square + 8
        else:
            square = move.to_square

        return board.piece_at(square), square

    def infer_last_move(self, previous_board, current_board):
        # brute force is fine here, just trying to reconstruct what happened
        current_fen = current_board.fen()
        for move in previous_board.legal_moves:
            temp = previous_board.copy(stack=False)
            temp.push(move)
            if temp.fen() == current_fen:
                return move
        return None

    def log_opponent_move(self, previous_board, current_board):
        if not self.verbose_tactics:
            return

        move = self.infer_last_move(previous_board, current_board)
        if move is None:
            return

        if move.promotion is not None:
            promoted = piece_names.get(move.promotion, "piece")
            print(f"[OPP-PROMOTION] {self.name}: opponent played {move.uci()} and promoted to {promoted}")

        captured, square = self.captured_piece(previous_board, move)
        if captured and captured.color == self.my_color and captured.piece_type != chess.KING:
            print(
                f"[LOST]    {self.name}: {piece_names[captured.piece_type]} on "
                f"{chess.square_name(square)} was captured"
            )

    def log_our_move(self, board, move):
        if not self.verbose_tactics:
            return

        # mostly just useful for debugging what the agent is doing
        if move.promotion is not None:
            promoted = piece_names.get(move.promotion, "piece")
            print(f"[PROMOTION] {self.name}: {move.uci()} promotes to {promoted}")

        if board.is_capture(move):
            captured, _ = self.captured_piece(board, move)
            if captured and captured.piece_type != chess.KING:
                print(f"[CAPTURE] {self.name}: {move.uci()} captures {piece_names[captured.piece_type]}")

        if board.gives_check(move):
            print(f"[CHECK]   {self.name}: {move.uci()}")

    def try_scholars_mate(self, board, legal_moves):
        # tiny opening hack because people still fall for this sometimes
        if not self.use_scholars_mate:
            return None
        if self.my_color != chess.WHITE or board.turn != chess.WHITE:
            return None
        if board.fullmove_number > 4:
            return None

        # if mate is already there, just take it
        if "h5f7" in legal_moves and board.gives_check(legal_moves["h5f7"]):
            return legal_moves["h5f7"]
        if "f3f7" in legal_moves and board.gives_check(legal_moves["f3f7"]):
            return legal_moves["f3f7"]

        # otherwise build toward the usual setup
        if board.fullmove_number == 1 and "e2e4" in legal_moves:
            return legal_moves["e2e4"]

        pawn_e4 = board.piece_at(chess.E4)
        queen_d1 = board.piece_at(chess.D1)
        bishop_f1 = board.piece_at(chess.F1)
        bishop_c4 = board.piece_at(chess.C4)

        if (
            pawn_e4
            and pawn_e4.color == chess.WHITE
            and bishop_f1
            and bishop_f1.color == chess.WHITE
            and "f1c4" in legal_moves
        ):
            return legal_moves["f1c4"]

        if (
            pawn_e4
            and pawn_e4.color == chess.WHITE
            and bishop_c4
            and bishop_c4.color == chess.WHITE
            and queen_d1
            and queen_d1.color == chess.WHITE
        ):
            if "d1h5" in legal_moves:
                return legal_moves["d1h5"]
            if "d1f3" in legal_moves:
                return legal_moves["d1f3"]

        return None

    def score_move(self, board, move):
        # this is just a lightweight ranking function, not real search
        side = board.turn
        moving_piece = board.piece_at(move.from_square)

        next_board = board.copy(stack=False)
        next_board.push(move)

        if next_board.is_checkmate():
            return 100000
        if next_board.is_stalemate():
            return -100000

        # don't blunder mate in 1 if i can avoid it
        for reply in next_board.legal_moves:
            reply_board = next_board.copy(stack=False)
            reply_board.push(reply)
            if reply_board.is_checkmate():
                return -90000

        score = 10 * self.material_balance(next_board, side)

        if board.gives_check(move):
            score += 25
        if board.is_capture(move):
            score += 15
        if move.promotion is not None:
            score += 35
        if next_board.can_claim_draw():
            score -= 30

        # adding a few endgame nudges so it doesn't play too passively
        is_endgame = len(board.piece_map()) <= 10
        if is_endgame and moving_piece is not None:
            if moving_piece.piece_type == chess.PAWN:
                start_rank = chess.square_rank(move.from_square)
                end_rank = chess.square_rank(move.to_square)
                advance = end_rank - start_rank if side == chess.WHITE else start_rank - end_rank
                score += 8 * advance

            if moving_piece.piece_type == chess.KING:
                f0 = chess.square_file(move.from_square)
                r0 = chess.square_rank(move.from_square)
                f1 = chess.square_file(move.to_square)
                r1 = chess.square_rank(move.to_square)

                d0 = abs(f0 - 3.5) + abs(r0 - 3.5)
                d1 = abs(f1 - 3.5) + abs(r1 - 3.5)
                score += 2 * (d0 - d1)

            if self.material_balance(board, side) > 0 and board.is_capture(move):
                score += 10

        # avoid repeating positions too much unless there is no better option
        score -= 40 * self.pos_counts.get(short_fen(next_board), 0)
        return score

    @torch.no_grad()
    def get_move(self, fen):
        try:
            board = chess.Board(fen)
            self.maybe_reset_game(board)

            if self.my_color is None:
                self.my_color = board.turn

            # if it's my turn again, the opponent must have just moved
            if self.last_after_move_board is not None and board.turn == self.my_color:
                self.log_opponent_move(self.last_after_move_board, board)

            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return None

            legal_map = {move.uci(): move for move in legal_moves}

            # always take mate in 1 if it's there
            for move in legal_moves:
                temp = board.copy(stack=False)
                temp.push(move)
                if temp.is_checkmate():
                    self.log_our_move(board, move)
                    self.last_after_move_board = temp
                    return move.uci()

            # small opening shortcut before asking the model
            scholar_move = self.try_scholars_mate(board, legal_map)
            if scholar_move is not None:
                self.log_our_move(board, scholar_move)
                temp = board.copy(stack=False)
                temp.push(scholar_move)
                self.last_after_move_board = temp
                return scholar_move.uci()

            key = short_fen(board)
            self.pos_counts[key] = self.pos_counts.get(key, 0) + 1

            prompt = self.build_prompt(fen)
            device = next(self.model.parameters()).device
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
            prompt_len = inputs["input_ids"].shape[1]

            candidates = []

            # try greedy first because sometimes the cleanest output is enough
            greedy = self.model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            greedy_text = self.tokenizer.decode(
                greedy[0][prompt_len:], skip_special_tokens=True
            ).strip()
            greedy_move = find_uci(greedy_text)
            if greedy_move in legal_map:
                candidates.append(legal_map[greedy_move])

            # then sample a few more in case greedy misses something useful
            if self.tries > 0:
                sampled = self.model.generate(
                    **inputs,
                    max_new_tokens=8,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    num_return_sequences=self.tries,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                for seq in sampled:
                    text = self.tokenizer.decode(
                        seq[prompt_len:], skip_special_tokens=True
                    ).strip()
                    uci = find_uci(text)
                    if uci in legal_map:
                        candidates.append(legal_map[uci])

            # if the model says nothing useful, just score all legal moves instead
            if not candidates:
                candidates = legal_moves

            unique_moves = {}
            for move in candidates:
                unique_moves[move.uci()] = move

            best_move = max(unique_moves.values(), key=lambda m: self.score_move(board, m))

            self.log_our_move(board, best_move)

            after = board.copy(stack=False)
            after.push(best_move)
            self.last_after_move_board = after

            return best_move.uci()

        except Exception:
            # fail safe so one bad generation doesn't kill the whole game
            return self.random_legal_move(fen)
