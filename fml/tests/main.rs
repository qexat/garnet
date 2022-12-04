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
fn test_unnamed_generic() {
    let src = include_str!("test_unnamed_generic.gt");
    let _output = fml::compile("test_unnamed_generic.gt", src);
}

#[test]
#[should_panic]
fn test_failure() {
    let src = include_str!("test_failure.gt");
    let _output = fml::compile("test_failure.gt", src);
}

#[test]
fn test_tuple1() {
    let src = include_str!("test_tuple1.gt");
    let _output = fml::compile("test_tuple1.gt", src);
}

#[test]
fn test_tuple2() {
    let src = include_str!("test_tuple2.gt");
    let _output = fml::compile("test_tuple2.gt", src);
}

#[test]
#[should_panic]
fn test_tuple3() {
    let src = include_str!("test_tuple3.gt");
    let _output = fml::compile("test_tuple3.gt", src);
}

#[test]
fn test_lambda1() {
    let src = include_str!("test_lambda1.gt");
    let _output = fml::compile("test_lambda1.gt", src);
}

#[test]
fn test_lambda2() {
    let src = include_str!("test_lambda2.gt");
    let _output = fml::compile("test_lambda2.gt", src);
}

#[test]
fn test_lambda3() {
    let src = include_str!("test_lambda3.gt");
    let _output = fml::compile("test_lambda3.gt", src);
}

#[test]
fn test_lambda4() {
    let src = include_str!("test_lambda4.gt");
    let _output = fml::compile("test_lambda4.gt", src);
}

#[test]
#[should_panic]
fn test_lambda5() {
    let src = include_str!("test_lambda5.gt");
    let _output = fml::compile("test_lambda5.gt", src);
}
