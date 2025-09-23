import csv
import math


def split_csv_into_three_files(input_csv, output_prefix):
    with open(input_csv, 'r', newline='', encoding='utf-8') as fin:
        reader = list(csv.reader(fin))
    header = reader[0]
    data_rows = reader[1:]
    header.insert(0, "line_no")
    total_rows = len(data_rows)
    chunk_size = math.ceil(total_rows / 3)

    print(f"Total data rows: {total_rows}")
    print(f"Chunk size (approx): {chunk_size}")

    new_data = []
    for i, row in enumerate(data_rows):
        line_no = i + 1
        new_data.append([line_no] + row)

    part1 = new_data[0:chunk_size]
    part2 = new_data[chunk_size:2*chunk_size]
    part3 = new_data[2*chunk_size:]

    # 6. Write each part into a separate CSV
    def write_part_file(part_rows, filename):
        with open(filename, 'w', newline='', encoding='utf-8') as fout:
            writer = csv.writer(fout)
            writer.writerow(header)
            writer.writerows(part_rows)

    write_part_file(part1, f"{output_prefix}_part1.csv")
    write_part_file(part2, f"{output_prefix}_part2.csv")
    write_part_file(part3, f"{output_prefix}_part3.csv")

    print("Done. Wrote three CSV files:" 
          f"\n  {output_prefix}_part1.csv"
          f"\n  {output_prefix}_part2.csv"
          f"\n  {output_prefix}_part3.csv")


if __name__ == "__main__":
    input_file = "../data_input/DiverSumm.csv"
    output_prefix = "DiverSum"
    split_csv_into_three_files(input_file, output_prefix)
