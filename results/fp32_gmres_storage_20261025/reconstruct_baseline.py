from pathlib import Path

root = Path(__file__).resolve().parents[2]
current = (root / "src/fp32_cuda_gmres.cu").read_text()
old = current
old = old.replace(
    "        Mem dr(fb(n)), dx(fb(n)), dinitial(fb(n)), dcandidate(fb(n)), dax(fb(n)), dpb(fb(n)),\n"
    "            dpa(fb(n)), dw(fb(n));",
    "        Mem dr(fb(n)), dx(fb(n)), dinitial(fb(n)), dcandidate(fb(n)), dax(fb(n)), dpb(fb(n)),\n"
    "            dpa(fb(n)), dw(fb(n)), basis(fb(n * (cap + 1)));",
)
start = old.index("        constexpr std::size_t storage_chunk = 16;")
end = old.index("        g[0] = beta;", start)
old = old[:start] + (
    "        e.copy(dw.f(), basis.f());\n"
    "        e.scale(finite(1.0f / beta, \"invalid FP32 CUDA normalization\"), basis.f());\n"
    "        std::vector<std::vector<float>> h(cap + 1, std::vector<float>(cap, 0));\n"
    "        std::vector<float> cs(cap, 0), sn(cap, 0), g(cap + 1, 0);\n"
) + old[end:]
start = old.index("            if (k == projected_capacity) {")
end = old.index("            float *vk =", start)
old = old[:start] + old[end:]
old = old.replace("basis_vector(k + 1)", "basis.f() + n * (k + 1)")
old = old.replace("basis_vector(k)", "basis.f() + n * k")
old = old.replace("basis_vector(j)", "basis.f() + n * j")
assert "basis_vector" not in old and "projected_capacity" not in old
target = Path(__file__).with_name("baseline_fp32_cuda_gmres.cu")
target.write_text(old)
print(target)
