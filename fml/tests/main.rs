use fml;

#[test]
fn test_thing() {
    let src = include_str!("test1.gt");
    let _output = fml::compile("test1.gt", src);
}

#[test]
fn test_thing2() {
    let src = include_str!("test2.gt");
    let _output = fml::compile("test2.gt", src);
}

#[test]
#[should_panic]
fn test_failure() {
    let src = include_str!("test_failure.gt");
    let _output = fml::compile("test_failure.gt", src);
}
