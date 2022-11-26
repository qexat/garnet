use fml;

#[test]
fn test_thing() {
    let src = include_str!("test1.gt");
    let _output = fml::compile("test1.gt", src);
}
