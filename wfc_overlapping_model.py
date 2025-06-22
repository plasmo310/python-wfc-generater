from __future__ import annotations

import os
import random
from itertools import chain
from typing import Dict, List, Tuple

from PIL import Image


class WFCSimpleOverlappingModel:
    """Wave Function Collapse モデル
    シンプルなOverlappingMoedelの実装

    入力画像からパターンを抽出し、指定されたサイズのグリッドに配置する
    回転考慮などの処理は含んでいない
    """

    # 画像の2次元リスト型
    ImageGridType = List[List[Tuple[int, int, int]]]

    # 許容されるパターンの隣接情報を保持する型
    # key: パターンインデックス, value: (左隣接パターン, 右隣接パターン, 上隣接パターン, 下隣接パターン)
    AllowedPattenAdjacenciesType = Dict[
        int, Tuple[List[int], List[int], List[int], List[int]]
    ]

    # 左, 右, 上, 下
    CHECK_DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    class TilePatternInfo:
        """
        タイルのパターン情報をまとめたクラス
        size x size のタイルパターンをvaluesとして1次元タプルで保持する

        例：3x3のパターンなら
        values = ((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))
        のようにRGB情報が9つ分入る形になる

        Args:
            patterns: パターン本体（1次元タプル）
            frequency: 出現頻度
            size: パターンの1辺サイズ
        """

        def __init__(
            self,
            patterns: Tuple[int, ...],
            frequency: int = 1,
            size: int = 3,
        ):
            self.values = patterns
            self.frequency = frequency
            self.size = size

        def right_except(self) -> Tuple[int, ...]:
            """右端以外の2列を1次元で返す"""
            return tuple(
                n for i, n in enumerate(self.values) if i % self.size != (self.size - 1)
            )

        def left_except(self) -> Tuple[int, ...]:
            """左端以外の2列を1次元で返す"""
            return tuple(n for i, n in enumerate(self.values) if i % self.size != 0)

        def bottom_except(self) -> Tuple[int, ...]:
            """下端以外の2行を1次元で返す"""
            return self.values[: (self.size * self.size) - self.size]

        def top_except(self) -> Tuple[int, ...]:
            """上端以外の2行を1次元で返す"""
            return self.values[self.size :]

    def __init__(
        self,
        input_image_grid: ImageGridType,
        check_pattern_size: int,
        output_grid_size: Tuple[int, int],
        use_input_pattern_frequency: bool = True,
    ):
        self._pattern_info_list = self._create_pattern_info_list(
            input_image_grid, check_pattern_size
        )
        self._check_pattern_size = check_pattern_size
        self._output_grid_size = output_grid_size
        self._use_input_pattern_frequency = use_input_pattern_frequency

        self._allowed_pattern_adjacencies_dict = (
            self._create_allowed_pattern_adjacencies_dict()
        )
        self._output_grid = self._initialize_grid()
        self._entropy_grid = self._initialize_entropy_grid()

    def _create_pattern_info_list(
        self, image_grid: ImageGridType, check_pattern_size: int
    ) -> List[TilePatternInfo]:
        """
        入力グリッドからユニークなパターンを抽出し、頻度もカウントする

        Args:
            image_grid (ImageGridType): 画像の2次元リスト
            check_pattern_size (int): チェックするパターンの1辺サイズ

        Returns:
            List[PatternInfo]: パターン情報のリスト
        """
        image_grid_count_x = len(image_grid[0])
        image_grid_count_y = len(image_grid)

        # 画像グリッドからパターンを抽出
        # 抽出範囲のサイズが画像グリッドのサイズを超えないようにする
        pattern_info_dict: Dict[str, WFCSimpleOverlappingModel.TilePatternInfo] = {}
        for y in range(image_grid_count_y - check_pattern_size + 1):
            for x in range(image_grid_count_x - check_pattern_size + 1):
                # 抽出範囲での全てのパターンを追加
                pattern_row_list = []
                for dy in range(check_pattern_size):
                    row = []
                    for dx in range(check_pattern_size):
                        ix = x + dx
                        iy = y + dy
                        row.append(image_grid[iy][ix])
                    pattern_row_list.append(tuple(row))
                # パターンの1次元タプルをキーとして値を設定
                pattern_values = tuple(chain.from_iterable(pattern_row_list))
                if pattern_values not in pattern_info_dict:
                    pattern_info_dict[pattern_values] = self.TilePatternInfo(
                        pattern_values,
                        frequency=1,
                        size=check_pattern_size,
                    )
                else:
                    # 重複していたらfrequencyとしてカウント
                    pattern_info_dict[pattern_values].frequency += 1
        return list(pattern_info_dict.values())

    def _create_allowed_pattern_adjacencies_dict(
        self,
    ) -> AllowedPattenAdjacenciesType:
        """
        各パターンごとに隣接可能なパターンインデックス辞書を作成する

        Returns:
            AllowedPattenAdjacenciesType: 隣接可能なパターンインデックスリストの辞書
        """
        patten_info_count = len(self._pattern_info_list)
        allowed_pattern_adjacencies_list = {
            i: ([], [], [], []) for i in range(patten_info_count)
        }

        # 各パターンの隣接可能なパターンを上下左右で探索
        for i in range(patten_info_count):
            pattern_i = self._pattern_info_list[i]
            for j in range(patten_info_count):
                pattern_j = self._pattern_info_list[j]
                # 左端以外と右端以外の2列が一致
                if pattern_i.left_except() == pattern_j.right_except():
                    allowed_pattern_adjacencies_list[i][1].append(j) # 右隣接
                    allowed_pattern_adjacencies_list[j][0].append(i) # 左隣接
                # 上端以外と下端以外の2行が一致
                if pattern_i.top_except() == pattern_j.bottom_except():
                    allowed_pattern_adjacencies_list[i][3].append(j) # 下隣接
                    allowed_pattern_adjacencies_list[j][2].append(i) # 上隣接
        return allowed_pattern_adjacencies_list

    def _initialize_grid(self) -> Dict[int, List[int]]:
        """出力グリッドを初期化する

        Returns:
            Dict[int, List[int]]: 出力グリッド（key: インデックス value: 許容されるパターンのリスト）
        """
        output_grid = {}
        for x in range(self._output_grid_size[0] * self._output_grid_size[1]):
            output_grid[x] = list(range(len(self._pattern_info_list)))
        return output_grid

    def _initialize_entropy_grid(self) -> Dict[int, int]:
        """エントロピー計算用グリッドを初期化する
        初期値は全てのセルでパターン数と同じ値に設定し、開始セルのエントロピーのみ1減らした状態にする

        Returns:
            Dict[int, int]: エントロピー計算用グリッド（key: インデックス value: エントロピー）
        """
        entropy_grid = {}
        for x in range(self._output_grid_size[0] * self._output_grid_size[1]):
            entropy_grid[x] = len(self._pattern_info_list)
        # 開始セルのエントロピーを1減らしておく
        start_index = random.randint(0, len(entropy_grid.keys()) - 1)
        entropy_grid[start_index] = len(self._pattern_info_list) - 1
        return entropy_grid

    def _get_lowest_entropy_cell_index(self) -> int:
        """エントロピーが最小のセルのインデックスを返す

        Returns:
            int: エントロピーが最小のセルのインデックス
        """
        return min(self._entropy_grid, key=self._entropy_grid.get)

    def _assign_random_pattern_to_cell(self, cell_index: int):
        """セルにパターンをランダムに1つ設定して確定する

        Args:
            cell_index (int): 対象セルのインデックス
        """
        # 該当セルで許容されているパターンを取得
        # 出現頻度に基づいてウェイトを設定する
        allowed_patterns = self._output_grid[cell_index]
        weights = [
            self._pattern_info_list[pattern_index].frequency
            for pattern_index in allowed_patterns
        ]
        pattern_index = random.choices(allowed_patterns, weights=weights)[0]
        # セルにパターンを適用してグリッドから削除
        self._output_grid[cell_index] = [pattern_index]
        del self._entropy_grid[cell_index]

    def _propagate_cells(self, cell_index: int) -> bool:
        """セルの確定を周囲に伝播してエントロピーを更新する
        セルの隣接パターン（上下左右）を全てチェックし、隣接セルの許容パターンを更新していく

        Args:
            cell_index (int): 対象セルのインデックス

        Returns:
            bool: 処理が成功したかどうか
        """
        to_update_cell_index_stack = {cell_index}

        while len(to_update_cell_index_stack) != 0:
            cell_index = to_update_cell_index_stack.pop()
            # セルの隣接パターン（上下左右）を全てチェック
            for direction, transform in enumerate(self.CHECK_DIRECTIONS):
                # 隣接セルのインデックスを計算
                neighbor_x = (
                    cell_index % self._output_grid_size[0] + transform[0]
                ) % self._output_grid_size[0]
                neighbor_y = (
                    cell_index // self._output_grid_size[0] + transform[1]
                ) % self._output_grid_size[1]
                neighbor_cell_index = (
                    neighbor_x + neighbor_y * self._output_grid_size[0]
                )
                # 隣接セルが未確定の場合
                if neighbor_cell_index in self._entropy_grid:
                    # セルの許容パターンを全て取得
                    allowed_pattern_indices_in_all_cell = set()
                    for pattern_index in self._output_grid[cell_index]:
                        allowed_pattern_indices: List[int] = (
                            self._allowed_pattern_adjacencies_dict[pattern_index][
                                direction
                            ]
                        )
                        for allowed_patten_index in allowed_pattern_indices:
                            allowed_pattern_indices_in_all_cell.add(allowed_patten_index)
                    # 隣接セルの許容パターンを更新
                    allowed_pattern_indices_in_neighbor_cell = self._output_grid[
                        neighbor_cell_index
                    ]
                    # 隣接セルのパターンがセルの許容パターンに含まれていない場合
                    # (セルの更新が発生して矛盾が発生した場合)
                    if not set(allowed_pattern_indices_in_neighbor_cell).issubset(
                        allowed_pattern_indices_in_all_cell
                    ):
                        # 隣接セルとセルに共通するパターンに絞り込んで設定する
                        shared_pattern_indices = [
                            x
                            for x in allowed_pattern_indices_in_neighbor_cell
                            if x in allowed_pattern_indices_in_all_cell
                        ]
                        if len(shared_pattern_indices) == 0:
                            return False
                        shared_pattern_indices.sort()
                        # 共通パターン数をエントロピーに設定する
                        self._output_grid[neighbor_cell_index] = shared_pattern_indices
                        self._entropy_grid[neighbor_cell_index] = len(
                            shared_pattern_indices
                        )
                        # 更に伝搬する
                        to_update_cell_index_stack.add(neighbor_cell_index)
        return True

    def run(self, random_seed=42) -> bool:
        """WFC処理の実行

        Args:
            random_seed (int, optional): 乱数シード値

        Returns:
            bool: 処理が成功したかどうか
        """
        # ランダムシードの設定
        random.seed(random_seed)

        # 処理の開始
        while True:
            # エントロピーグリッドが全て空になったら処理終了
            if len(self._entropy_grid) == 0:
                return True
            # エントロピーが最小のセルを取得し、
            # 許容されているパターンからランダムに1つ選んで設定する
            lowest_entropy_cell = self._get_lowest_entropy_cell_index()
            self._assign_random_pattern_to_cell(lowest_entropy_cell)
            # セルの確定を周囲に伝播してエントロピーを更新する
            is_success = self._propagate_cells(lowest_entropy_cell)
            if not is_success:
                break
        return False

    def get_result_image_grid(self) -> ImageGridType:
        """WFC処理の結果を2次元リストとして返す

        Returns:
            ImageGridType: WFC処理の結果
        """
        result_image_grid: WFCSimpleOverlappingModel.ImageGridType = []
        for y in range(self._output_grid_size[1]):
            row = []
            for x in range(self._output_grid_size[0]):
                idx = x + y * self._output_grid_size[0]
                val = next(iter(self._output_grid[idx]))
                row.append(self._pattern_info_list[val].values[0])
            result_image_grid.append(row)
        return result_image_grid


def load_image_grid(image_path: str) -> WFCSimpleOverlappingModel.ImageGridType:
    """
    画像ファイルを読み込んで2次元リストとして返却する

    Args:
        image_path (str): 画像ファイルのパス

    Returns:
        ImageGridType: 画像の2次元リスト
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    pixels = list(image.getdata())
    image_grid = [pixels[y * width : (y + 1) * width] for y in range(height)]
    return image_grid


def save_image_from_grid(
    image_grid: WFCSimpleOverlappingModel.ImageGridType, output_path: str
) -> None:
    """
    2次元リストを画像として保存する

    Args:
        image_grid (ImageGridType): 画像の2次元リスト
        output_path (str): 保存先のファイルパス
    """
    height = len(image_grid)
    width = len(image_grid[0])
    image = Image.new("RGB", (width, height))
    flat_pixels = [pixel for row in image_grid for pixel in row]
    image.putdata(flat_pixels)
    image.save(output_path)


def main():
    # 入出力情報
    OUTPUT_IMAGE_NAME_LIST = [
        "tile_patterns.png",
        "road_small.png",
        "wakame.png"
    ]
    INPUT_DIR_NAME = "images/input"
    OUTPUT_DIR_NAME = "images/output_wfc"

    # 乱数情報
    INIT_RANDOM_SEED = 42  # 乱数シード値
    MAX_RETRY_COUNT = 10 # 最大リトライ数

    # WFCパラメータ
    CHECK_PATTERN_SIZE = 3  # チェックするパターンの1辺サイズ
    OUTPUT_GRID_SIZE = (240, 240)  # 出力グリッドサイズ
    USE_INPUT_PATTERN_FREQUENCY = True  # 入力パターンの頻度を使用するかどうか

    for image_name in OUTPUT_IMAGE_NAME_LIST:
        # 入出力ファイルパス
        input_image_path = os.path.join(
            os.path.dirname(__file__), INPUT_DIR_NAME, image_name
        )
        output_image_path = os.path.join(
            os.path.dirname(__file__), OUTPUT_DIR_NAME, image_name
        )

        # 乱数シード値を変えながら数回実行
        for i in range(MAX_RETRY_COUNT):
            # 画像情報の読み込み
            input_image_grid = load_image_grid(input_image_path)
            input_random_seed = INIT_RANDOM_SEED + i * 100

            # WFCモデルの初期化と実行
            wfc_model = WFCSimpleOverlappingModel(
                input_image_grid=input_image_grid,
                check_pattern_size=CHECK_PATTERN_SIZE,
                output_grid_size=OUTPUT_GRID_SIZE,
                use_input_pattern_frequency=USE_INPUT_PATTERN_FREQUENCY,
            )
            is_success = wfc_model.run(
                random_seed=input_random_seed
            )
            if not is_success:
                print(f"Random seed: {input_random_seed} => Failed to create image: '{image_name}'.")
                continue

            # 結果の画像グリッドを取得して保存
            output_image_grid = wfc_model.get_result_image_grid()
            save_image_from_grid(output_image_grid, output_image_path)
            print(f"Random seed: {input_random_seed} => Complete save image: {output_image_path}.")
            break


if __name__ == "__main__":
    main()
