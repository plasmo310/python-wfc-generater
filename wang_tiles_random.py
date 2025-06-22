from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List

from PIL import Image


class WangTilesUtility:
    """WangTiles関連処理"""

    # 下記論文を参考にした4x4のタイルIDパターン
    # https://graphics.stanford.edu/papers/tile_mapping_gh2004/
    TILE_ID_STR_PATTERN_LIST = [
        "0010", "0110", "0111", "0011",
        "1010", "1110", "1111", "1011",
        "1000", "1100", "1101", "1001",
        "0000", "0100", "0101", "0001",
    ]

    @dataclass
    class TileId:
        """タイルIDデータクラス"""

        top: int
        right: int
        bottom: int
        left: int

        def get_id_str(self) -> str:
            return f"{self.top}{self.right}{self.bottom}{self.left}"

    @staticmethod
    def get_tile_id_pattern_list() -> List[WangTilesUtility.TileId]:
        """タイルIDパターンリストを取得する

        Returns:
            List[WangTilesUtility.TileId]: タイルIDパターンリスト
        """
        tile_id_list = []
        for s in WangTilesUtility.TILE_ID_STR_PATTERN_LIST:
            tile_id_list.append(
                WangTilesUtility.TileId(
                    top=int(s[0]), right=int(s[1]), bottom=int(s[2]), left=int(s[3])
                )
            )
        return tile_id_list

    @staticmethod
    def tile_id_to_index(tile_id: WangTilesUtility.TileId) -> int:
        """タイルIDからインデックスに変換する

        Args:
            tile_id (WangTilesUtility.TileId): タイルID

        Returns:
            int: タイルIDのインデックス
        """
        return WangTilesUtility.TILE_ID_STR_PATTERN_LIST.index(tile_id.get_id_str())

    @staticmethod
    def create_random_tile_grids(
        tile_count_x: int,
        tile_count_y: int,
        tile_id_pattern_list: List[TileId],
        random_seed: int = 0,
    ) -> List[List[WangTilesUtility.TileId]]:
        """指定された情報からランダムにタイルIDのGrid情報を生成する

        Args:
            tile_count_x (int): X方向のタイル数
            tile_count_y (int): Y方向のタイル数
            tile_id_pattern_list (List[TileId]): タイルIDパターンリスト
            random_seed (int, optional): 乱数シード値. Defaults to 0.

        Returns:
            List[List[WangTilesUtility.TileId]]: タイルIDのGrid情報
        """
        random.seed(random_seed)

        # grid初期化
        id_tile_list: List[List[WangTilesUtility.TileId]] = [
            [None for _ in range(tile_count_x)] for _ in range(tile_count_y)
        ]

        # タイルを左辺、上辺どちらかが繋がるようにランダムに抽出
        for y in range(tile_count_y):
            for x in range(tile_count_x):
                filter_pattern_id_list = []
                # 最初のタイルはランダムに選択
                if x == 0 and y == 0:
                    id_tile_list[y][x] = random.choice(tile_id_pattern_list)
                    continue
                # 左辺 == 左タイルの右辺
                left_neighbor_tile_id = id_tile_list[x][y - 1]
                if left_neighbor_tile_id:
                    for tile_id in tile_id_pattern_list:
                        if tile_id.left == left_neighbor_tile_id.right:
                            filter_pattern_id_list.append(tile_id)
                # 上辺 == 上タイルの下辺
                top_neighbor_tile_id = id_tile_list[x - 1][y]
                if top_neighbor_tile_id:
                    for tile_id in tile_id_pattern_list:
                        if tile_id.top == top_neighbor_tile_id.bottom:
                            filter_pattern_id_list.append(tile_id)

                id_tile_list[x][y] = random.choice(filter_pattern_id_list)

        return id_tile_list

    def generate_image_from_tile_grid(
        tile_grids: List[List[WangTilesUtility.TileId]],
        tile_size: int,
        tile_image_path: str,
        output_image_path: str,
    ) -> Image.Image:
        """tile_gridから画像を生成する

        Args:
            tile_grid (List[List[WangTilesUtility.TileId]]): tile_idの2次元リスト
            tile_size (int, optional): タイル一つ辺りのサイズ.
            tile_image_path (str): tile画像のパス
            output_image_path (str): 生成する画像のパス
        """
        # タイル画像読み込み
        tile_image = Image.open(tile_image_path)
        tile_count_x = tile_image.width // tile_size
        tile_count_y = tile_image.height // tile_size

        # tile_gridから画像を生成する
        grid_count_y = len(tile_grids)
        grid_count_x = len(tile_grids[0])
        result_image = Image.new(
            "RGBA", (tile_size * grid_count_x, tile_size * grid_count_y)
        )
        for y, row in enumerate(tile_grids):
            for x, tile_id in enumerate(row):
                tile_index = WangTilesUtility.tile_id_to_index(tile_id)
                src_x = (tile_index % tile_count_x) * tile_size
                src_y = (tile_index // tile_count_y) * tile_size
                tile_crop = tile_image.crop(
                    (src_x, src_y, src_x + tile_size, src_y + tile_size)
                )
                result_image.paste(tile_crop, (x * tile_size, y * tile_size))
        result_image.save(output_image_path)


def main():
    # タイル画像情報
    INPUT_TILE_IMAGE_PATH = os.path.join(
        os.path.dirname(__file__), "images/input/tile_patterns.png"
    )
    INPUT_TILE_SIZE = 24

    # 生成画像情報
    OUTPUT_IMAGE_PATH = os.path.join(
        os.path.dirname(__file__), "output_wang_tiles/tile_patterns.png"
    )
    OUTPUT_TILE_COUNT = 10

    # タイルIDパターンリストを取得
    tile_id_pattern_list = WangTilesUtility.get_tile_id_pattern_list()

    # ランダムにタイルIDのGrid情報を生成
    random_tile_grids = WangTilesUtility.create_random_tile_grids(
        tile_count_x=OUTPUT_TILE_COUNT,
        tile_count_y=OUTPUT_TILE_COUNT,
        tile_id_pattern_list=tile_id_pattern_list,
        random_seed=3,
    )

    # 作成したGrid情報からタイル画像を生成
    WangTilesUtility.generate_image_from_tile_grid(
        tile_grids=random_tile_grids,
        tile_size=INPUT_TILE_SIZE,
        tile_image_path=INPUT_TILE_IMAGE_PATH,
        output_image_path=OUTPUT_IMAGE_PATH,
    )


if __name__ == "__main__":
    main()
