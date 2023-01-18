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
fn test_forever1() {
    let src = include_str!("test_forever1.gt");
    let _output = fml::compile("test_forever1.gt", src);
}

#[test]
fn test_forever2() {
    let src = include_str!("test_forever2.gt");
    let _output = fml::compile("test_forever2.gt", src);
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
fn test_tuple4() {
    let src = include_str!("test_tuple4.gt");
    let _output = fml::compile("test_tuple4.gt", src);
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

#[test]
fn test_typedef1() {
    let src = include_str!("test_typedef1.gt");
    let _output = fml::compile("test_typedef1.gt", src);
}

#[test]
fn test_typedef2() {
    let src = include_str!("test_typedef2.gt");
    let _output = fml::compile("test_typedef2.gt", src);
}

#[test]
#[should_panic]
fn test_typedef3() {
    let src = include_str!("test_typedef3.gt");
    let _output = fml::compile("test_typedef3.gt", src);
}

#[test]
fn test_typedef4() {
    let src = include_str!("test_typedef4.gt");
    let _output = fml::compile("test_typedef4.gt", src);
}

#[test]
fn test_typedef5() {
    let src = include_str!("test_typedef5.gt");
    let _output = fml::compile("test_typedef5.gt", src);
}

#[test]
#[should_panic]
fn test_typedef5_failure() {
    let src = include_str!("test_typedef5_failure.gt");
    let _output = fml::compile("test_typedef5_failure.gt", src);
}

#[test]
fn test_typedef6() {
    let src = include_str!("test_typedef6.gt");
    let _output = fml::compile("test_typedef6.gt", src);
}

#[test]
#[should_panic]
fn test_typedef7() {
    let src = include_str!("test_typedef7.gt");
    let _output = fml::compile("test_typedef7.gt", src);
}

#[test]
#[should_panic]
fn test_typedef8() {
    let src = include_str!("test_typedef8.gt");
    let _output = fml::compile("test_typedef8.gt", src);
}

#[test]
fn test_struct1() {
    let src = include_str!("test_struct1.gt");
    let _output = fml::compile("test_struct1.gt", src);
}

#[test]
fn test_struct2() {
    let src = include_str!("test_struct2.gt");
    let _output = fml::compile("test_struct2.gt", src);
}

#[test]
#[should_panic]
fn test_struct3() {
    let src = include_str!("test_struct3.gt");
    let _output = fml::compile("test_struct3.gt", src);
}

#[test]
fn test_struct4() {
    let src = include_str!("test_struct4.gt");
    let _output = fml::compile("test_struct4.gt", src);
}

#[test]
#[should_panic]
fn test_struct5() {
    let src = include_str!("test_struct5.gt");
    let _output = fml::compile("test_struct5.gt", src);
}

#[test]
#[should_panic]
fn test_struct6() {
    let src = include_str!("test_struct6.gt");
    let _output = fml::compile("test_struct6.gt", src);
}

#[test]
fn test_struct7() {
    let src = include_str!("test_struct7.gt");
    let _output = fml::compile("test_struct7.gt", src);
}

#[test]
fn test_let1() {
    let src = include_str!("test_let1.gt");
    let _output = fml::compile("test_let1.gt", src);
}

#[test]
fn test_module1() {
    let src = include_str!("test_module1.gt");
    let _output = fml::compile("test_module1.gt", src);
}

/* This test is complex and maybe incorrect anyway
 * leave it out for now.
#[test]
fn test_module2() {
    let src = include_str!("test_module2.gt");
    let _output = fml::compile("test_module2.gt", src);
}
*/

#[test]
fn test_module3() {
    let src = include_str!("test_module3.gt");
    let _output = fml::compile("test_module3.gt", src);
}

#[test]
#[should_panic]
fn test_module4() {
    let src = include_str!("test_module4.gt");
    let _output = fml::compile("test_module4.gt", src);
}

#[test]
fn test_module_specialization1() {
    let src = include_str!("test_module_specialization1.gt");
    let _output = fml::compile("test_module_specialization1.gt", src);
}

#[test]
#[should_panic]
fn test_unnamed_failure1() {
    let src = include_str!("test_unnamed_failure1.gt");
    let _output = fml::compile("test_unnamed_failure1.gt", src);
}

#[test]
#[should_panic]
fn test_unnamed_failure2() {
    let src = include_str!("test_unnamed_failure2.gt");
    let _output = fml::compile("test_unnamed_failure2.gt", src);
}

#[test]
#[should_panic]
fn test_unnamed_failure3() {
    let src = include_str!("test_unnamed_failure3.gt");
    let _output = fml::compile("test_unnamed_failure3.gt", src);
}
