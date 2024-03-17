pub mod embedding;
pub mod server;

fn main() {
    let config_url = String::from("https://magpidev9b7b.blob.core.windows.net/onnx-files/onnx-test/config.json?sp=r&st=2024-03-16T12:59:14Z&se=2024-03-16T20:59:14Z&spr=https&sv=2022-11-02&sr=b&sig=dXuJYVhZbC4tWCuknVAW8CuaQm2X7fAf%2Fh41SjQCpU8%3D");
    let tokenizer_url = String::from("https://magpidev9b7b.blob.core.windows.net/onnx-files/onnx-test/tokenizer.json?sp=r&st=2024-03-16T13:01:44Z&se=2024-03-16T21:01:44Z&spr=https&sv=2022-11-02&sr=b&sig=KyDZz9Orm58Xo%2Fb7EPbynM3bogaUPR0mcikrcC6z1AI%3D");
    let special_tokens_map_url = String::from("https://magpidev9b7b.blob.core.windows.net/onnx-files/onnx-test/special_tokens_map.json?sp=r&st=2024-03-16T13:01:13Z&se=2024-03-16T21:01:13Z&spr=https&sv=2022-11-02&sr=b&sig=TLvUj91TAM1Ex8NDWJYAlOQ94PzDqbbEp4BdluFXV8k%3D");
    let tokenizer_config_url = String::from("https://magpidev9b7b.blob.core.windows.net/onnx-files/onnx-test/tokenizer_config.json?sp=r&st=2024-03-16T13:01:31Z&se=2024-03-16T21:01:31Z&spr=https&sv=2022-11-02&sr=b&sig=4aaN9lGgAnbnXgme1e5b4UiM1YvdRcpurTb50wgWaQU%3D");
    let onnx_url = String::from("https://magpidev9b7b.blob.core.windows.net/onnx-files/onnx-test/model.onnx?sp=r&st=2024-03-16T13:00:53Z&se=2024-03-16T21:00:53Z&spr=https&sv=2022-11-02&sr=b&sig=cyiDzpdpat0%2FjJcDPf7aUmvAoEIHd5lmOqsXFh9ynkk%3D");
    server::run::start_server(None, embedding::ModelSource::HuggingFace);
}
