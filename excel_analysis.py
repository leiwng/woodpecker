from openpyxl import load_workbook, Workbook
import pandas as pd


if __name__ == "__main__":

    excel_fp = r"E:\染色体数据\240314_羊水质控分析\羊水质控分析表格_WZ_leiw_240314.xlsx"
    df = pd.read_excel(excel_fp, sheet_name="Original")
    new_df = pd.DataFrame(columns=df.columns)

    for index, row in df.iterrows():

        new_row = row.copy()

        if "标本分割排列错误情况" in row and row["标本分割排列错误情况"] != "":
            # 获取"标本分割排列错误情况"列的值
            error_situation = row["标本分割排列错误情况"]

            # 将错误情况拆分成多行
            if isinstance(error_situation, str) and "，" in error_situation:
                errors = error_situation.split("，")

                error_situation = error_situation.strip()
                # 去除字符串中间的空格
                error_situation = error_situation.replace(" ", "")

                # 对每个拆分后的错误情况创建一行，并添加到新的DataFrame中
                for error in errors:
                    # 更新"标本分割排列错误情况"列的值为当前拆分后的错误情况
                    clean_error = error.strip()
                    # 去除字符串中间的空格
                    clean_error = clean_error.replace(" ", "")
                    clean_error_parts = clean_error.split("、")
                    clean_error_parts_len = len(clean_error_parts)
                    if (
                        clean_error[-1] == "反" or clean_error[-5:] == "反（遮挡）"
                    ) and clean_error_parts_len > 2:
                        for item in clean_error_parts:
                            new_row["标本分割排列错误情况"] = f"{clean_error}-{item}"
                            # 错误个数为1
                            new_row["错误个数"] = 1
                            # 添加新行到新的DataFrame中
                            new_df = pd.concat(
                                [new_df, pd.DataFrame([new_row])], ignore_index=True
                            )
                    elif (
                        clean_error[-1] == "反" or clean_error[-5:] == "反（遮挡）"
                    ) and clean_error_parts_len == 1:
                        new_row["标本分割排列错误情况"] = clean_error
                        # 错误个数为1
                        new_row["错误个数"] = 1
                        new_row["错误类型"] = "极性"
                        # 添加新行到新的DataFrame中
                        new_df = pd.concat(
                            [new_df, pd.DataFrame([new_row])], ignore_index=True
                        )
                    else:
                        new_row["标本分割排列错误情况"] = clean_error
                        # 错误个数为1
                        new_row["错误个数"] = 1
                        # 添加新行到新的DataFrame中
                        new_df = pd.concat(
                            [new_df, pd.DataFrame([new_row])], ignore_index=True
                        )
            elif isinstance(error_situation, str) and "、" in error_situation:

                error_situation = error_situation.strip()
                # 去除字符串中间的空格
                error_situation = error_situation.replace(" ", "")

                # 更新"标本分割排列错误情况"列的值为当前拆分后的错误情况
                clean_error = error_situation.strip()
                # 去除字符串中间的空格
                clean_error = clean_error.replace(" ", "")
                clean_error_parts = clean_error.split("、")
                clean_error_parts_len = len(clean_error_parts)
                if (
                    clean_error[-1] == "反" or clean_error[-5:] == "反（遮挡）"
                ) and clean_error_parts_len > 2:
                    for item in clean_error_parts:
                        new_row["标本分割排列错误情况"] = f"{clean_error}-{item}"
                        # 错误个数为1
                        new_row["错误个数"] = 1
                        # 添加新行到新的DataFrame中
                        new_df = pd.concat(
                            [new_df, pd.DataFrame([new_row])], ignore_index=True
                        )
                elif (
                    clean_error[-1] == "反" or clean_error[-5:] == "反（遮挡）"
                ) and clean_error_parts_len == 1:
                    new_row["标本分割排列错误情况"] = clean_error
                    # 错误个数为1
                    new_row["错误个数"] = 1
                    new_row["错误类型"] = "极性"
                    # 添加新行到新的DataFrame中
                    new_df = pd.concat(
                        [new_df, pd.DataFrame([new_row])], ignore_index=True
                    )
                else:
                    new_row["标本分割排列错误情况"] = clean_error
                    # 错误个数为1
                    new_row["错误个数"] = 1
                    # 添加新行到新的DataFrame中
                    new_df = pd.concat(
                        [new_df, pd.DataFrame([new_row])], ignore_index=True
                    )
            elif isinstance(error_situation, str) and (
                error_situation[-1] == "反" or error_situation[-5:] == "反（遮挡）"
            ):

                error_situation = error_situation.strip()
                # 去除字符串中间的空格
                error_situation = error_situation.replace(" ", "")

                new_row["标本分割排列错误情况"] = error_situation
                # 错误个数为1
                new_row["错误个数"] = 1
                new_row["错误类型"] = "极性"
                # 添加新行到新的DataFrame中
                new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
            else:
                new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            # 如果不需要处理的列，直接将原始行添加到新的DataFrame中
            # new_df = new_df.append(row, ignore_index=True)
            new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)

    # 将新的DataFrame保存为Excel文件
    output_excel_fp = (
        r"E:\染色体数据\240314_羊水质控分析\羊水质控分析表格_Details_leiw_240314.xlsx"
    )
    new_df.to_excel(output_excel_fp, index=False)
