def extract_unique_error_lines(input_file, output_file):
    seen = set()
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:

        for line in f_in:
            if "warning:" in line.lower():
                if line not in seen:
                    seen.add(line)
                    f_out.write(line)

if __name__ == "__main__":
    extract_unique_error_lines("install_log.txt", "warning_unique.txt")
1